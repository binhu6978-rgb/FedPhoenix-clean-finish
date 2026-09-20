param(
    [Parameter(Mandatory = $true)]
    [int]$BaselineProcessId
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\admin\.conda\envs\fd\python.exe"
$logDirectory = Join-Path $workspace "results\formal_beta03_logs"
$baselineSummary = Join-Path $workspace (
    "results\formal_beta03_metrics\" +
    "cifar10_resnet18_FedPhoenix_seed1_paper_beta03_seed1_1200_summary.json"
)

Wait-Process -Id $BaselineProcessId

if (-not (Test-Path -LiteralPath $baselineSummary)) {
    throw "FedPhoenix baseline ended without producing its summary: $baselineSummary"
}

$arguments = @(
    "-u",
    "-X", "utf8",
    "main_fed.py",
    "--dataset", "cifar10",
    "--model", "resnet18",
    "--algorithm", "FedPhoenixRecovery",
    "--recovery_mode", "joint",
    "--epochs", "200",
    "--num_users", "100",
    "--frac", "0.1",
    "--local_ep", "5",
    "--local_bs", "50",
    "--bs", "256",
    "--lr", "0.01",
    "--momentum", "0.5",
    "--weight_decay", "0",
    "--iid", "0",
    "--noniid_case", "5",
    "--data_beta", "0.3",
    "--FP_conv", "1000",
    "--FP_fc", "0",
    "--reset", "0.015625",
    "--remethod", "ori_normal",
    "--seed", "1",
    "--generate_data", "0",
    "--validation_samples", "1000",
    "--eval_every", "5",
    "--recovery_pool_multiplier", "3",
    "--recovery_probe_samples", "64",
    "--metrics_log_dir", "results\formal_beta03_metrics",
    "--recovery_log_dir", "results\formal_beta03_matching",
    "--run_name", "recovery_joint_beta03_seed1_200",
    "--gpu", "0"
)

$recoveryProcess = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $workspace `
    -RedirectStandardOutput (
        Join-Path $logDirectory "recovery_joint_seed1.stdout.log"
    ) `
    -RedirectStandardError (
        Join-Path $logDirectory "recovery_joint_seed1.stderr.log"
    ) `
    -WindowStyle Hidden `
    -Wait `
    -PassThru

if ($recoveryProcess.ExitCode -ne 0) {
    throw "FedPhoenixRecovery exited with code $($recoveryProcess.ExitCode)"
}
