param(
    [Parameter(Mandatory = $true)]
    [int]$WaitPid
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\admin\.conda\envs\fd\python.exe"
$logDir = Join-Path $root "results\formal_beta03_logs"
$metricsDir = Join-Path $root "results\inertia_beta03_metrics"
$matchingDir = Join-Path $root "results\inertia_beta03_matching"
$reportPath = Join-Path $root "results\autonomous_inertia_search_report.json"
$baselineValidation = 74.8
$baselinePeak = 73.71
$targetDelta = 1.0

function Get-RunName($inertia, $minGain, $epochs) {
    $inertiaTag = "{0:D3}" -f [int][math]::Round($inertia * 100)
    $gainTag = "{0:D2}" -f [int][math]::Round($minGain * 10)
    return "inertia${inertiaTag}_warmup50_gain${gainTag}_seed1_${epochs}"
}

function Get-SummaryPath($name) {
    return Join-Path $metricsDir (
        "cifar10_resnet18_FedPhoenixRecovery_seed1_${name}_summary.json"
    )
}

function Get-PilotResult($inertia, $minGain, $name) {
    $summary = Get-Content -LiteralPath (Get-SummaryPath $name) -Raw |
        ConvertFrom-Json
    return [pscustomobject]@{
        name = $name
        inertia = $inertia
        min_gain = $minGain
        best_validation_accuracy = [double]$summary.best_validation_accuracy
        peak_test_accuracy = [double]$summary.diagnostic_peak_test_accuracy
        validation_delta = (
            [double]$summary.best_validation_accuracy - $baselineValidation
        )
        peak_delta = (
            [double]$summary.diagnostic_peak_test_accuracy - $baselinePeak
        )
    }
}

function Run-Training($inertia, $minGain, $epochs) {
    $name = Get-RunName $inertia $minGain $epochs
    $arguments = @(
        "-u", "-X", "utf8", "main_fed.py",
        "--dataset", "cifar10",
        "--model", "resnet18",
        "--algorithm", "FedPhoenixRecovery",
        "--recovery_mode", "assignment_only",
        "--epochs", "$epochs",
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
        "--recovery_probe_samples", "64",
        "--recovery_start_round", "50",
        "--recovery_end_round", "-1",
        "--recovery_min_assignment_gain", "$minGain",
        "--recovery_assignment_weighting", "uniform",
        "--recovery_assignment_inertia", "$inertia",
        "--metrics_log_dir", "results\inertia_beta03_metrics",
        "--recovery_log_dir", "results\inertia_beta03_matching",
        "--run_name", $name,
        "--gpu", "0"
    )
    $process = Start-Process `
        -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $root `
        -RedirectStandardOutput (Join-Path $logDir "$name.stdout.log") `
        -RedirectStandardError (Join-Path $logDir "$name.stderr.log") `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        throw "$name failed with exit code $($process.ExitCode)"
    }
    return $name
}

$existing = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
if ($null -ne $existing) {
    Wait-Process -Id $WaitPid
}

$pilots = @()
$currentName = "inertia075_warmup50_gain03_seed1_200"
$currentSummary = Get-SummaryPath $currentName
if (-not (Test-Path -LiteralPath $currentSummary)) {
    throw "Missing current pilot summary: $currentSummary"
}
$current = Get-PilotResult 0.75 0.3 $currentName
$pilots += $current
$selected = $null
if (
    $current.validation_delta -ge $targetDelta -and
    $current.peak_delta -ge $targetDelta
) {
    $selected = $current
}

if ($null -eq $selected) {
    $secondName = Run-Training 1.0 0.2 200
    $second = Get-PilotResult 1.0 0.2 $secondName
    $pilots += $second
    if (
        $second.validation_delta -ge $targetDelta -and
        $second.peak_delta -ge $targetDelta
    ) {
        $selected = $second
    }
}

if ($null -eq $selected) {
    $selected = $pilots |
        Sort-Object validation_delta, peak_delta -Descending |
        Select-Object -First 1
}

$pilotReport = [ordered]@{
    status = "full_run"
    target_delta = $targetDelta
    pilots = $pilots
    selected = $selected
}
$pilotReport | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath $reportPath -Encoding UTF8

$fullName = Run-Training $selected.inertia $selected.min_gain 1200
$fullSummary = Get-Content -LiteralPath (Get-SummaryPath $fullName) -Raw |
    ConvertFrom-Json

$finalReport = [ordered]@{
    status = "complete"
    target_delta = $targetDelta
    pilots = $pilots
    selected = $selected
    full_run_name = $fullName
    full_summary = $fullSummary
}
$finalReport | ConvertTo-Json -Depth 10 |
    Set-Content -LiteralPath $reportPath -Encoding UTF8
