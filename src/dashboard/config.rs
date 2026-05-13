use clap::Args;

/// Args specific to the `dashboard` (and `all`) subcommand. Worker-level
/// config still flows in via the top-level [`crate::Config`] flatten.
#[derive(Args, Debug, Clone)]
pub struct DashboardArgs {
    /// Address to bind the HTTP server to.
    #[arg(long, env = "DASHBOARD_BIND", default_value = "127.0.0.1:7788")]
    pub bind: String,

    /// Skip the `Secure` flag on session cookies (local development only —
    /// browsers refuse Secure cookies on plain HTTP).
    #[arg(long, env = "DASHBOARD_INSECURE", default_value_t = false)]
    pub insecure: bool,
}
