fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    // Set build timestamp
    println!(
        "cargo:rustc-env=BUILD_TIMESTAMP={}",
        std::env::var("SOURCE_DATE_EPOCH")
            .map(|epoch| {
                use std::time::{SystemTime, UNIX_EPOCH};
                let timestamp = epoch.parse::<u64>().unwrap();
                let datetime = UNIX_EPOCH + std::time::Duration::from_secs(timestamp);
                format!("{:?}", datetime)
            })
            .unwrap_or_else(|_| format!("{:?}", std::time::SystemTime::now()))
    );
}