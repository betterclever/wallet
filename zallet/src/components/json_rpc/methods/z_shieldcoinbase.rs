//! Implementation of the `z_shieldcoinbase` RPC method.

use std::convert::Infallible;
use std::future::Future;

use documented::Documented;
use jsonrpsee::core::{JsonValue, RpcResult};
use schemars::JsonSchema;
use secrecy::ExposeSecret;
use serde::Serialize;
use transparent::address::TransparentAddress;
use uuid::Uuid;
use zaino_state::FetchServiceSubscriber;
use zcash_address::ZcashAddress;
use zcash_client_backend::{
    data_api::{
        Account as _, InputSource, TransparentOutputFilter, WalletRead,
        wallet::{
            ConfirmationsPolicy, SpendingKeys, TargetHeight, create_proposed_transactions,
            input_selection::GreedyInputSelector, propose_shielding_coinbase,
        },
    },
    fees::StandardFeeRule,
    proposal::Proposal,
    wallet::OvkPolicy,
};
use zcash_client_sqlite::AccountUuid;
use zcash_keys::{address::Address, keys::UnifiedSpendingKey};
use zcash_proofs::prover::LocalTxProver;
use zcash_protocol::value::Zatoshis;

use crate::{
    components::{
        database::{DbConnection, DbHandle},
        json_rpc::{
            asyncop::{ContextInfo, OperationId},
            payments::{PrivacyPolicy, SendResult, broadcast_transactions, parse_memo},
            server::LegacyCode,
            utils::{JsonZec, value_from_zatoshis},
        },
        keystore::KeyStore,
    },
    prelude::*,
};

#[cfg(feature = "transparent-key-import")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "transparent-key-import")]
use zcash_client_backend::wallet::TransparentAddressSource;
#[cfg(feature = "transparent-key-import")]
use zcash_script::script;

/// The result of a `z_shieldcoinbase` pre-flight call.
///
/// Mirrors the JSON object returned by `zcashd`'s `z_shieldcoinbase`:
#[derive(Clone, Debug, Serialize, Documented, JsonSchema)]
pub(crate) struct ShieldCoinbaseResult {
    /// Number of coinbase UTXOs that were eligible for shielding but were not
    /// selected by this operation. Non-zero when the caller supplied a `limit`
    /// that was smaller than the count of eligible coinbase UTXOs.
    #[serde(rename = "remainingUTXOs")]
    remaining_utxos: u64,

    /// Total value (in ZEC) of coinbase UTXOs that were eligible for
    /// shielding but were not selected by this operation. See `remainingUTXOs`.
    #[serde(rename = "remainingValue")]
    remaining_value: JsonZec,

    /// Number of coinbase UTXOs being shielded by this operation.
    #[serde(rename = "shieldingUTXOs")]
    shielding_utxos: u64,

    /// Total value (in ZEC) of coinbase UTXOs being shielded by this
    /// operation.
    #[serde(rename = "shieldingValue")]
    shielding_value: JsonZec,

    /// Operation id to pass to `z_getoperationstatus` /
    /// `z_getoperationresult` to retrieve the final result.
    opid: OperationId,
}

impl ShieldCoinbaseResult {
    /// Combines the synchronously-computed [`Preflight`] numerics with the
    /// [`OperationId`] returned by [`crate::components::json_rpc::asyncop`]
    /// once the async portion is registered.
    pub(super) fn new(preflight: Preflight, opid: OperationId) -> Self {
        Self {
            remaining_utxos: preflight.remaining_utxos,
            remaining_value: preflight.remaining_value,
            shielding_utxos: preflight.shielding_utxos,
            shielding_value: preflight.shielding_value,
            opid,
        }
    }
}

pub(crate) type ResultType = ShieldCoinbaseResult;
pub(crate) type Response = RpcResult<ResultType>;

/// Pre-flight numeric fields, computed before the async portion runs.
///
/// Held as a separate type so that the [`OperationId`] (only available after
/// the async operation is registered) can be joined with these values to
/// produce the final [`ShieldCoinbaseResult`].
pub(crate) struct Preflight {
    pub(super) remaining_utxos: u64,
    pub(super) remaining_value: JsonZec,
    pub(super) shielding_utxos: u64,
    pub(super) shielding_value: JsonZec,
}

pub(super) const PARAM_FROMADDRESS_DESC: &str = "Source of coinbase UTXOs to shield. Either a single transparent address owned by this \
     wallet, or an account UUID to sweep every coinbase UTXO across that account's transparent \
     receivers.";
pub(super) const PARAM_TOADDRESS_DESC: &str = "Any Zcash shielded address (Sapling, Orchard, or Unified with a shielded receiver) that \
     will receive the shielded funds. Need not belong to this wallet. Transparent or TEX \
     destinations are rejected.";
pub(super) const PARAM_FEE_DESC: &str =
    "If provided, must be null. Zallet always calculates and applies the ZIP-317 fee internally.";
pub(super) const PARAM_LIMIT_DESC: &str = "If supplied, caps the number of selected coinbase UTXOs to the highest-value `n` of those \
     eligible. Recommended for wallets with many eligible coinbase UTXOs: without it, a single \
     transaction is built containing all eligible UTXOs, which can exceed transaction-size \
     limits at broadcast time.";
pub(super) const PARAM_MEMO_DESC: &str = "If supplied, stored in the memo field of the resulting shielded payment. Must be a \
     hex-encoded string (up to 1024 hex characters = 512 bytes).";
pub(super) const PARAM_PRIVACY_POLICY_DESC: &str = "If set, it must be null.";

/// Soft threshold above which [`call`] emits a `warn!` log about potential
/// transaction-size issues at broadcast time.
pub(super) const COINBASE_INPUTS_WARN_THRESHOLD: u64 = 400;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn call(
    mut wallet: DbHandle,
    keystore: KeyStore,
    chain: FetchServiceSubscriber,
    fromaddress: String,
    toaddress: String,
    fee: Option<JsonValue>,
    limit: Option<u32>,
    memo: Option<String>,
    privacy_policy: Option<String>,
) -> RpcResult<(
    Preflight,
    Option<ContextInfo>,
    impl Future<Output = RpcResult<SendResult>>,
)> {
    // `fee` and `privacy_policy` exist for positional compatibility with
    // `zcashd`'s `z_shieldcoinbase` only — Zallet always computes the fee
    // internally and always shields under a fixed privacy policy. Reject any
    // non-null value so callers can't silently believe they're configuring
    // either.
    if fee.is_some() {
        return Err(LegacyCode::InvalidParameter
            .with_static("Zallet always calculates fees internally; the fee field must be null."));
    }
    if privacy_policy.is_some() {
        return Err(LegacyCode::InvalidParameter.with_static(
            "Zallet does not support configuring the privacy policy for z_shieldcoinbase; \
             the privacy_policy field must be null.",
        ));
    }

    // Parse the destination address.
    let to_zcash_address: ZcashAddress = toaddress.parse().map_err(|_| {
        LegacyCode::InvalidParameter.with_message(format!(
            "Invalid parameter, unknown address format: {toaddress}"
        ))
    })?;

    // Parse the memo parameter (hex-encoded).
    let memo = memo.as_deref().map(parse_memo).transpose()?;
    let limit_usize = limit.map(|n| n as usize);

    // Resolve `fromaddress` to the source account + its source transparent
    // addresses (one address for a t-addr input, all account receivers for a
    // UUID input).
    let (account_id, from_addrs) = resolve_fromaddress(wallet.as_ref(), &fromaddress)?;

    if from_addrs.is_empty() {
        return Err(LegacyCode::InvalidParameter
            .with_static("No source transparent addresses resolved from `fromaddress`."));
    }

    let account = wallet
        .get_account(account_id)
        .map_err(|e| LegacyCode::Database.with_message(e.to_string()))?
        .ok_or_else(|| {
            LegacyCode::Database.with_message(format!("Account vanished mid-call: {account_id:?}"))
        })?;

    let params = *wallet.params();
    let input_selector = GreedyInputSelector::new();

    // Create the shielding proposal. Uses Zatoshis::ZERO as the shielding
    // threshold to shield all available coinbase UTXOs (or all up to `limit`
    // when supplied). `propose_shielding_coinbase` hard-codes
    // `TransparentOutputFilter::CoinbaseOnly`, attaches the supplied memo to
    // the resulting shielded payment, and produces no transparent or shielded
    // change (preserving the privacy invariant that a shielded change output
    // would let `toaddress` learn the sender's total selected-coinbase value).
    let proposal = propose_shielding_coinbase::<_, _, _, _, Infallible>(
        wallet.as_mut(),
        &params,
        &input_selector,
        &StandardFeeRule::Zip317,
        Zatoshis::ZERO,
        &from_addrs,
        to_zcash_address,
        memo,
        limit_usize,
    )
    .map_err(|e| {
        LegacyCode::Wallet.with_message(format!("Failed to propose shielding transaction: {e}"))
    })?;

    // The shielding operation always reveals the transparent sender(s). When the caller asks us to
    // sweep multiple source addresses (i.e. via an account UUID expanded to >1 receivers), the
    // resulting transaction also links those addresses on-chain, so we must additionally allow
    // that.
    crate::components::json_rpc::payments::enforce_privacy_policy(
        &proposal,
        PrivacyPolicy::AllowLinkingAccountAddresses,
    )?;

    // Pre-flight numerics
    let (shielding_utxos, shielding_value_zats) = sum_selected_inputs(&proposal)?;
    let target_height = proposal.min_target_height();
    let (total_utxos, total_value_zats) =
        enumerate_eligible(wallet.as_mut(), &from_addrs, target_height)?;

    let remaining_utxos = total_utxos.checked_sub(shielding_utxos).ok_or_else(|| {
        LegacyCode::Wallet.with_static(
            "Internal accounting error: proposal selected more UTXOs than \
             enumeration found (likely a chain race during shielding setup).",
        )
    })?;
    let remaining_value_zats = (total_value_zats - shielding_value_zats).ok_or_else(|| {
        LegacyCode::Wallet.with_static(
            "Internal accounting error: proposal value exceeds enumerated total \
             (likely a chain race during shielding setup).",
        )
    })?;

    if shielding_utxos > COINBASE_INPUTS_WARN_THRESHOLD {
        warn!(
            "z_shieldcoinbase: proposal selected {} coinbase UTXOs, which exceeds the \
             soft warning threshold of {}. The resulting transaction may exceed \
             network/mempool size limits at broadcast time. If broadcast fails, retry \
             with a `limit` parameter to shield in smaller batches.",
            shielding_utxos, COINBASE_INPUTS_WARN_THRESHOLD,
        );
    }

    let preflight = Preflight {
        remaining_utxos,
        remaining_value: value_from_zatoshis(remaining_value_zats),
        shielding_utxos,
        shielding_value: value_from_zatoshis(shielding_value_zats),
    };

    // Derive the spending key for the source account.
    let derivation = account.source().key_derivation().ok_or_else(|| {
        LegacyCode::InvalidAddressOrKey.with_message(format!(
            "No payment source found for account {}.",
            account_id.expose_uuid(),
        ))
    })?;
    let seed = keystore
        .decrypt_seed(derivation.seed_fingerprint())
        .await
        .map_err(|e| match e.kind() {
            crate::error::ErrorKind::Generic if e.to_string() == "Wallet is locked" => {
                LegacyCode::WalletUnlockNeeded.with_message(e.to_string())
            }
            _ => LegacyCode::Database.with_message(e.to_string()),
        })?;
    let usk = UnifiedSpendingKey::from_seed(
        wallet.params(),
        seed.expose_secret(),
        derivation.account_index(),
    )
    .map_err(|e| LegacyCode::InvalidAddressOrKey.with_message(e.to_string()))?;

    #[cfg(feature = "transparent-key-import")]
    let standalone_keys =
        collect_standalone_keys(wallet.as_mut(), &keystore, account_id, &proposal).await?;

    Ok((
        preflight,
        Some(ContextInfo::new(
            "z_shieldcoinbase",
            serde_json::json!({
                "fromaddress": fromaddress,
                "toaddress": toaddress,
                "limit": limit,
            }),
        )),
        run(
            wallet,
            chain,
            proposal,
            SpendingKeys::new(
                usk,
                #[cfg(feature = "transparent-key-import")]
                standalone_keys,
            ),
        ),
    ))
}

/// Classified form of the `fromaddress` parameter. Pure-function output of
/// [`parse_fromaddress`], split out from the DB-touching
/// [`resolve_fromaddress`] so the parsing rules can be unit-tested.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum FromAddressInput {
    /// An account UUID: sweep every transparent receiver in this account.
    AccountUuid(Uuid),
    /// A wallet-owned transparent address: shield coinbase UTXOs at exactly
    /// this address.
    TransparentAddress(TransparentAddress),
}

/// Classify the string form of `fromaddress` as either an account UUID or a
/// transparent address, without touching the wallet database.
///
/// UUIDs and Zcash transparent addresses use disjoint string forms (UUIDs are
/// hex with dashes at fixed positions; transparent addresses are base58check
/// with a leading `t1` / `t2` / `t3` / `tm`), so the parse order is safe.
pub(super) fn parse_fromaddress(
    params: &crate::network::Network,
    fromaddress: &str,
) -> RpcResult<FromAddressInput> {
    if let Ok(uuid) = Uuid::parse_str(fromaddress) {
        return Ok(FromAddressInput::AccountUuid(uuid));
    }
    match Address::decode(params, fromaddress) {
        Some(Address::Transparent(addr)) => Ok(FromAddressInput::TransparentAddress(addr)),
        Some(_) => Err(LegacyCode::InvalidAddressOrKey.with_message(format!(
            "Invalid `fromaddress`: only transparent addresses are accepted (got a \
             non-transparent Zcash address): {fromaddress}",
        ))),
        None => Err(LegacyCode::InvalidParameter.with_message(format!(
            "Invalid `fromaddress`: expected a wallet-owned transparent address or an \
             account UUID, got {fromaddress:?}",
        ))),
    }
}

/// Resolve `fromaddress` to a source account and the transparent addresses
/// to draw coinbase UTXOs from.
///
/// Accepted shapes:
/// * a wallet-owned transparent address — selects only that address.
/// * an account UUID — selects every transparent receiver in that account.
fn resolve_fromaddress(
    wallet: &DbConnection,
    fromaddress: &str,
) -> RpcResult<(AccountUuid, Vec<TransparentAddress>)> {
    match parse_fromaddress(wallet.params(), fromaddress)? {
        FromAddressInput::AccountUuid(uuid) => {
            let account_id = AccountUuid::from_uuid(uuid);
            if wallet
                .get_account(account_id)
                .map_err(|e| LegacyCode::Database.with_message(e.to_string()))?
                .is_none()
            {
                return Err(LegacyCode::InvalidParameter
                    .with_message(format!("Unknown account UUID: {fromaddress}")));
            }
            let from_addrs = wallet
                .get_transparent_receivers(account_id, true, true)
                .map_err(|e| LegacyCode::Database.with_message(e.to_string()))?
                .into_keys()
                .collect();
            Ok((account_id, from_addrs))
        }
        FromAddressInput::TransparentAddress(addr) => {
            let owner = wallet
                .find_account_for_address(wallet.params(), &Address::Transparent(addr))
                .map_err(|e| LegacyCode::Database.with_message(e.to_string()))?
                .ok_or_else(|| {
                    LegacyCode::InvalidAddressOrKey.with_message(format!(
                        "Transparent address is not owned by any account in this wallet: \
                         {fromaddress}",
                    ))
                })?;
            Ok((owner, vec![addr]))
        }
    }
}

fn sum_selected_inputs(
    proposal: &Proposal<StandardFeeRule, Infallible>,
) -> RpcResult<(u64, Zatoshis)> {
    let mut count: u64 = 0;
    let mut sum = Zatoshis::ZERO;
    for step in proposal.steps() {
        for utxo in step.transparent_inputs() {
            count = count.saturating_add(1);
            sum = (sum + utxo.value()).ok_or_else(|| {
                LegacyCode::Wallet
                    .with_static("Internal error: shielding value sum overflowed Zatoshis bounds.")
            })?;
        }
    }
    Ok((count, sum))
}

fn enumerate_eligible(
    wallet: &mut DbConnection,
    from_addrs: &[TransparentAddress],
    target_height: TargetHeight,
) -> RpcResult<(u64, Zatoshis)> {
    let mut total_utxos: u64 = 0;
    let mut total_value_zats = Zatoshis::ZERO;
    for addr in from_addrs {
        let utxos = wallet
            .get_spendable_transparent_outputs(
                addr,
                target_height,
                ConfirmationsPolicy::MIN,
                TransparentOutputFilter::CoinbaseOnly,
            )
            .map_err(|e| LegacyCode::Database.with_message(e.to_string()))?;
        total_utxos = total_utxos.saturating_add(utxos.len() as u64);
        for utxo in utxos {
            total_value_zats = (total_value_zats + utxo.value()).ok_or_else(|| {
                LegacyCode::Wallet.with_static(
                    "Internal error: total transparent value overflowed Zatoshis bounds.",
                )
            })?;
        }
    }
    Ok((total_utxos, total_value_zats))
}

#[cfg(feature = "transparent-key-import")]
async fn collect_standalone_keys(
    wallet: &mut DbConnection,
    keystore: &KeyStore,
    account_id: AccountUuid,
    proposal: &Proposal<StandardFeeRule, Infallible>,
) -> RpcResult<HashMap<TransparentAddress, Vec<secp256k1::SecretKey>>> {
    let standalone_addrs: HashSet<TransparentAddress> = wallet
        .get_transparent_receivers(account_id, true, true)
        .map_err(|e| LegacyCode::Database.with_message(e.to_string()))?
        .into_iter()
        .filter_map(|(addr, metadata)| match metadata.source() {
            TransparentAddressSource::StandalonePubkey(_)
            | TransparentAddressSource::StandaloneScript(_) => Some(addr),
            TransparentAddressSource::Derived { .. } => None,
        })
        .collect();

    let mut keys: HashMap<TransparentAddress, Vec<secp256k1::SecretKey>> = HashMap::new();
    for step in proposal.steps() {
        for input in step.transparent_inputs() {
            if let Some(address) = script::FromChain::parse(&input.txout().script_pubkey().0)
                .ok()
                .as_ref()
                .and_then(TransparentAddress::from_script_from_chain)
            {
                if !standalone_addrs.contains(&address) {
                    continue;
                }
                let secret_key = keystore
                    .decrypt_standalone_transparent_key(&address)
                    .await
                    .map_err(|e| match e.kind() {
                        crate::error::ErrorKind::Generic if e.to_string() == "Wallet is locked" => {
                            LegacyCode::WalletUnlockNeeded.with_message(e.to_string())
                        }
                        _ => LegacyCode::Database.with_message(e.to_string()),
                    })?;
                keys.entry(address).or_default().push(secret_key);
            }
        }
    }
    Ok(keys)
}

/// Construct and broadcast the shielding transaction.
async fn run(
    mut wallet: DbHandle,
    chain: FetchServiceSubscriber,
    proposal: Proposal<StandardFeeRule, Infallible>,
    spending_keys: SpendingKeys,
) -> RpcResult<SendResult> {
    let prover = LocalTxProver::bundled();
    let (wallet, txids) = crate::spawn_blocking!("z_shieldcoinbase runner", move || {
        let params = *wallet.params();
        create_proposed_transactions::<_, _, Infallible, _, Infallible, _>(
            wallet.as_mut(),
            &params,
            &prover,
            &prover,
            &spending_keys,
            OvkPolicy::Sender,
            &proposal,
        )
        .map(|txids| (wallet, txids))
    })
    .await
    .map_err(|e| {
        LegacyCode::Wallet.with_message(format!("Failed to build shielding transaction: {e}"))
    })?
    .map_err(|e| {
        LegacyCode::Wallet.with_message(format!("Failed to build shielding transaction: {e}"))
    })?;

    broadcast_transactions(&wallet, chain, txids.into()).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::json_rpc::server::LegacyCode;
    use crate::network::Network;
    use zcash_protocol::consensus;

    fn mainnet() -> Network {
        Network::Consensus(consensus::Network::MainNetwork)
    }

    fn testnet() -> Network {
        Network::Consensus(consensus::Network::TestNetwork)
    }

    // Reused from validate_address.rs tests.
    const MAINNET_P2PKH: &str = "t1VydNnkjBzfL1iAMyUbwGKJAF7PgvuCfMY";
    const MAINNET_P2SH: &str = "t3Vz22vK5z2LcKEdg16Yv4FFneEL1zg9ojd";
    const TESTNET_P2PKH: &str = "tmGqwWtL7RsbxikDSN26gsbicxVr2xJNe86";
    // Reused from validate_address.rs:160.
    const MAINNET_SAPLING: &str =
        "zs1z7rejlpsa98s2rrrfkwmaxu53e4ue0ulcrw0h4x5g8jl04tak0d3mm47vdtahatqrlkngh9slya";

    const VALID_UUID: &str = "123e4567-e89b-12d3-a456-426614174000";

    #[test]
    fn uuid_classified_as_account_uuid() {
        let parsed = parse_fromaddress(&mainnet(), VALID_UUID).unwrap();
        let expected_uuid = Uuid::parse_str(VALID_UUID).unwrap();
        assert_eq!(parsed, FromAddressInput::AccountUuid(expected_uuid));
    }

    #[test]
    fn p2pkh_taddr_classified_as_transparent_address() {
        let parsed = parse_fromaddress(&mainnet(), MAINNET_P2PKH).unwrap();
        match parsed {
            FromAddressInput::TransparentAddress(TransparentAddress::PublicKeyHash(_)) => {}
            other => panic!("Expected P2PKH TransparentAddress, got {other:?}"),
        }
    }

    #[test]
    fn p2sh_taddr_classified_as_transparent_address() {
        let parsed = parse_fromaddress(&mainnet(), MAINNET_P2SH).unwrap();
        match parsed {
            FromAddressInput::TransparentAddress(TransparentAddress::ScriptHash(_)) => {}
            other => panic!("Expected P2SH TransparentAddress, got {other:?}"),
        }
    }

    #[test]
    fn testnet_taddr_classified_on_testnet() {
        let parsed = parse_fromaddress(&testnet(), TESTNET_P2PKH).unwrap();
        match parsed {
            FromAddressInput::TransparentAddress(TransparentAddress::PublicKeyHash(_)) => {}
            other => panic!("Expected testnet P2PKH TransparentAddress, got {other:?}"),
        }
    }

    #[test]
    fn mainnet_taddr_on_testnet_rejected() {
        // Wrong network: mainnet t1 address parsed under testnet params is
        // not a valid transparent address, so it should fail the parser with
        // an InvalidParameter rather than an InvalidAddressOrKey.
        let err = parse_fromaddress(&testnet(), MAINNET_P2PKH).unwrap_err();
        assert_eq!(err.code(), LegacyCode::InvalidParameter as i32);
    }

    #[test]
    fn sapling_shielded_address_rejected_as_non_transparent() {
        let err = parse_fromaddress(&mainnet(), MAINNET_SAPLING).unwrap_err();
        assert_eq!(err.code(), LegacyCode::InvalidAddressOrKey as i32);
    }

    #[test]
    fn garbage_string_rejected() {
        let err = parse_fromaddress(&mainnet(), "not-a-thing").unwrap_err();
        assert_eq!(err.code(), LegacyCode::InvalidParameter as i32);
    }

    #[test]
    fn empty_string_rejected() {
        let err = parse_fromaddress(&mainnet(), "").unwrap_err();
        assert_eq!(err.code(), LegacyCode::InvalidParameter as i32);
    }
}
