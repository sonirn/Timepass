use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use anyhow::{Context, Result};
use clap::Parser;
use dashmap::DashSet;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use bip39::{Language, Mnemonic};
use bitcoin::bip32::{Xpriv as ExtendedPrivKey, DerivationPath};
use bitcoin::Network;
use bitcoin::secp256k1::{PublicKey, SecretKey, Secp256k1};
use ethers::utils::keccak256;

#[derive(Parser, Debug, Clone)]
struct Cli {
    #[clap(long, default_value = "Wordlist.txt")]
    wordlist: PathBuf,
    
    #[clap(long, default_value = "eth.txt")]
    addresses: PathBuf,
    
    #[clap(long, default_value = "found.txt")]
    output: PathBuf,
    
    #[clap(long, default_value = "checkpoint.json")]
    checkpoint: PathBuf,
    
    #[clap(
        long, 
        use_value_delimiter = true, 
        value_delimiter = ',', 
        default_value = "m/44'/60'/0'/0/0"
    )]
    derivation_paths: Vec<String>,
    
    #[clap(long, default_value = "10000")]
    chunk_size: usize,
    
    #[clap(long, default_value = "0")]
    threads: usize,
    
    #[clap(long, default_value = "300")]
    checkpoint_interval: u64,
    
    #[clap(long)]
    encryption_password: Option<String>,
    
    #[clap(short, long)]
    verbose: bool,
}

#[derive(Serialize, Deserialize, Clone)]
struct CheckpointState {
    last_processed_index: u64,
    total_processed: u64,
    found_results: Vec<FoundResult>,
}

#[derive(Serialize, Deserialize, Clone)]
struct FoundResult {
    passphrase: String,
    address: String,
    derivation_path: String,
}

#[derive(Clone)]
struct BruteForceContext {
    words: Arc<Vec<String>>,
    addresses: Arc<DashSet<String>>,
    derivation_paths: Arc<Vec<String>>,
    found_results: Arc<Mutex<Vec<FoundResult>>>,
    options: Arc<Cli>,
}

struct CombinationIterator {
    n: u32,
    k: u32,
    current: Option<Vec<u32>>,
    #[allow(dead_code)]
    skip_count: u64,
}


impl CombinationIterator {
    fn new(n: u32, k: u32, skip: u64) -> Self {
        if k > n {
            return Self {
                n,
                k,
                current: None,
                skip_count: 0,
            };
        }
        
        let mut current = Vec::with_capacity(k as usize);
        for i in 0..k {
            current.push(i);
        }
        
        let mut iter = Self {
            n,
            k,
            current: Some(current),
            skip_count: skip,
        };
        
        for _ in 0..skip {
            if !iter.advance() {
                break;
            }
        }
        
        iter
    }

    fn advance(&mut self) -> bool {
        let curr = match &mut self.current {
            Some(c) => c,
            None => return false,
        };
        
        let mut i = self.k as usize - 1;
        while i > 0 && curr[i] == self.n - self.k + i as u32 {
            i -= 1;
        }
        
        if curr[0] == self.n - self.k && i == 0 {
            self.current = None;
            return false;
        }
        
        curr[i] += 1;
        for j in i + 1..self.k as usize {
            curr[j] = curr[j - 1] + 1;
        }
        
        true
    }
}

impl Iterator for CombinationIterator {
    type Item = Vec<u32>;
    
    fn next(&mut self) -> Option<Self::Item> {
        let result = self.current.clone();
        if result.is_some() {
            self.advance();
        }
        result
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .context("Failed to configure thread pool")?;
    }
    
    let words = read_wordlist(&cli.wordlist).context("Failed to read wordlist")?;
    let addresses_vec = read_addresses(&cli.addresses).context("Failed to read addresses")?;
    
    if words.len() != 24 {
        return Err(anyhow::anyhow!(
            "Wordlist must contain exactly 24 words, found {}",
            words.len()
        ));
    }
    
    let ctx = Arc::new(BruteForceContext {
        words: Arc::new(words),
        addresses: Arc::new(DashSet::from_iter(addresses_vec)),
        derivation_paths: Arc::new(cli.derivation_paths.clone()),
        found_results: Arc::new(Mutex::new(Vec::new())),
        options: Arc::new(cli.clone()),
    });
    
    let checkpoint = load_checkpoint(&cli.checkpoint)?;
    let start_index = checkpoint.as_ref().map_or(0, |cp| cp.last_processed_index);
    
    if let Some(cp) = &checkpoint {
        let mut locked = ctx.found_results.lock().unwrap();
        locked.extend(cp.found_results.clone());
    }
    
    let total_combinations = calculate_combinations(24, 12);
    let progress = Arc::new(AtomicU64::new(
        checkpoint.as_ref().map_or(0, |cp| cp.total_processed)
    ));
    
    let pb = ProgressBar::new(total_combinations);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("##-"),
    );
    pb.set_position(progress.load(Ordering::Relaxed));
    
    let start_time = Instant::now();
    
    {
        let pb_clone = pb.clone();
        let progress_clone = progress.clone();
        std::thread::spawn(move || {
            update_progress(pb_clone, progress_clone, start_time);
        });
    }
    
    {
        let ctx_clone = ctx.clone();
        let progress_clone = progress.clone();
        let cp_path = cli.checkpoint.clone();
        let cp_interval = cli.checkpoint_interval;
        std::thread::spawn(move || {
            periodic_checkpoint(ctx_clone, progress_clone, cp_path, cp_interval);
        });
    }
    
    process_combinations(ctx.clone(), progress.clone(), start_index, cli.chunk_size, &pb)?;
    
    save_checkpoint(
        &cli.checkpoint,
        total_combinations,
        progress.load(Ordering::Relaxed),
        &ctx.found_results.lock().unwrap(),
    )?;
    
    pb.finish_and_clear();
    Ok(())
}

fn process_combinations(
    ctx: Arc<BruteForceContext>,
    progress: Arc<AtomicU64>,
    start_index: u64,
    chunk_size: usize,
    pb: &ProgressBar,
) -> Result<()> {
    let mut combo_iter = CombinationIterator::new(24, 12, start_index);
    let mut processed = 0;
    let total_remaining = calculate_combinations(24, 12) - start_index;
    
    while processed < total_remaining {
        let chunk: Vec<_> = combo_iter.by_ref().take(chunk_size).collect();
        if chunk.is_empty() {
            break;
        }
        
        let chunk_len = chunk.len();
        
        chunk.into_par_iter().for_each(|indices| {
            let passphrase: Vec<String> = indices
                .iter()
                .map(|&idx| ctx.words[idx as usize].clone())
                .collect();
                
            for path in ctx.derivation_paths.iter() {
                if let Some(addr) = derive_eth_address(&passphrase, path) {
                    if ctx.addresses.contains(&addr.to_lowercase()) {
                        let joined_pass = passphrase.join(" ");
                        let found = FoundResult {
                            passphrase: joined_pass.clone(),
                            address: addr.clone(),
                            derivation_path: path.clone(),
                        };
                        
                        {
                            let mut locked = ctx.found_results.lock().unwrap();
                            locked.push(found.clone());
                        }
                        
                        if let Err(e) = append_found(&found, &ctx.options) {
                            eprintln!("Could not append found passphrase: {}", e);
                        }
                    }
                }
            }
            
            progress.fetch_add(1, Ordering::Relaxed);
        });
        
        processed += chunk_len as u64;
        pb.set_position(progress.load(Ordering::Relaxed));
    }
    
    Ok(())
}

fn derive_eth_address(passphrase: &[String], derivation_path: &str) -> Option<String> {
    let phrase = passphrase.join(" ");
    let mnemonic = Mnemonic::parse_in_normalized(Language::English, &phrase).ok()?;
    let seed_bytes = mnemonic.to_seed("");
    
    let dp = DerivationPath::from_str(derivation_path).ok()?;
    let secp = Secp256k1::new();
    let master_key = ExtendedPrivKey::new_master(Network::Bitcoin, &seed_bytes).ok()?;
    let child_key = master_key.derive_priv(&secp, &dp).ok()?;
    
    let mut private_key = child_key.private_key.secret_bytes().to_vec();
    let eth_addr = eth_address_from_private_key(&private_key);
    private_key.zeroize();
    
    eth_addr.map(|a| format!("0x{}", hex::encode(a)))
}

fn eth_address_from_private_key(pk_bytes: &[u8]) -> Option<[u8; 20]> {
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(pk_bytes).ok()?;
    let pk = PublicKey::from_secret_key(&secp, &sk);
    let uncompressed = pk.serialize_uncompressed();
    let pubkey_bytes = &uncompressed[1..];
    let hash = keccak256(pubkey_bytes);
    
    let mut address = [0u8; 20];
    address.copy_from_slice(&hash[12..32]);
    Some(address)
}

fn append_found(result: &FoundResult, options: &Cli) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&options.output)
        .context("Failed to open output file")?;
        
    let line = format!(
        "{} => {} (path: {})\n",
        result.passphrase, result.address, result.derivation_path
    );
    
    if let Some(pw) = &options.encryption_password {
        let encrypted = encrypt_string(&line, pw)?;
        writeln!(file, "{encrypted}")?;
    } else {
        write!(file, "{line}")?;
    }
    
    Ok(())
}

fn encrypt_string(data: &str, password: &str) -> Result<String> {
    use base64::{engine::general_purpose, Engine as _};
    let combined = format!("{password}:{data}");
    let encoded = general_purpose::STANDARD.encode(combined);
    Ok(format!("ENCRYPTED:{encoded}"))
}

fn periodic_checkpoint(
    ctx: Arc<BruteForceContext>,
    progress: Arc<AtomicU64>,
    checkpoint_path: PathBuf,
    interval_secs: u64,
) {
    loop {
        std::thread::sleep(Duration::from_secs(interval_secs));
        let current_progress = progress.load(Ordering::Relaxed);
        let found_results = ctx.found_results.lock().unwrap();
        
        match save_checkpoint(
            &checkpoint_path,
            current_progress,
            current_progress,
            &found_results,
        ) {
            Err(e) => eprintln!("Failed to save checkpoint: {e}"),
            _ => {}
        }
        
        if current_progress >= calculate_combinations(24, 12) {
            break;
        }
    }
}

fn save_checkpoint(
    path: &PathBuf,
    last_index: u64,
    total_processed: u64,
    found_results: &Vec<FoundResult>,
) -> Result<()> {
    let state = CheckpointState {
        last_processed_index: last_index,
        total_processed,
        found_results: found_results.clone(),
    };
    
    let json_str = serde_json::to_string(&state).context("Failed to serialize checkpoint")?;
    let temp_path = path.with_extension("tmp");
    std::fs::write(&temp_path, json_str).context("Checkpoint data write failed")?;
    std::fs::rename(&temp_path, path).context("Renaming checkpoint file failed")?;
    
    Ok(())
}

fn load_checkpoint(path: &PathBuf) -> Result<Option<CheckpointState>> {
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let result = serde_json::from_str(&content).context("Failed to parse checkpoint")?;
            Ok(Some(result))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow::Error::new(e).context("Failed reading checkpoint file")),
    }
}

fn read_wordlist(path: &PathBuf) -> Result<Vec<String>> {
    let file = File::open(path).with_context(|| format!("Could not open Wordlist: {path:?}"))?;
    let reader = BufReader::new(file);
    
    let words: Vec<String> = reader
        .lines()
        .filter_map(|line| {
            if let Ok(s) = line {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
            None
        })
        .collect();
        
    Ok(words)
}

fn read_addresses(path: &PathBuf) -> Result<Vec<String>> {
    let file = File::open(path).with_context(|| format!("Could not open addresses: {path:?}"))?;
    let reader = BufReader::new(file);
    
    let addresses: Vec<String> = reader
        .lines()
        .filter_map(|line| {
            if let Ok(s) = line {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    let addr = if trimmed.starts_with("0x") {
                        trimmed.to_string()
                    } else {
                        format!("0x{}", trimmed)
                    };
                    return Some(addr.to_lowercase());
                }
            }
            None
        })
        .collect();
        
    Ok(addresses)
}

fn calculate_combinations(n: u64, k: u64) -> u64 {
    let k = if k > n - k { n - k } else { k };
    let mut result = 1u64;
    
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    
    result
}

fn update_progress(pb: ProgressBar, progress: Arc<AtomicU64>, start_time: Instant) {
    let mut last_count = 0u64;
    let mut last_time = Instant::now();
    
    loop {
        std::thread::sleep(Duration::from_secs(1));
        let current_count = progress.load(Ordering::Relaxed);
        let now = Instant::now();
        let dt = now.duration_since(last_time).as_secs_f64();
        let pass_s = (current_count - last_count) as f64 / dt;
        let total_elapsed = start_time.elapsed().as_secs_f64();
        let pass_avg = if total_elapsed > 0.0 {
            current_count as f64 / total_elapsed
        } else {
            0.0
        };
        
        pb.set_message(format!("{:.2} pass/s (avg: {:.2})", pass_s, pass_avg));
        pb.set_position(current_count);
        last_count = current_count;
        last_time = now;
        
        if current_count >= pb.length().unwrap_or(0) {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_combination_iterator() {
        let mut iter = CombinationIterator::new(4, 2, 0);
        let combos: Vec<_> = iter.collect();
        assert_eq!(combos.len(), 6);
        assert_eq!(combos[0], vec![0, 1]);
        assert_eq!(combos[5], vec![2, 3]);
    }
  }
