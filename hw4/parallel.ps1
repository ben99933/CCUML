# run.ps1
$startTime = Get-Date
$pythonPath = "$env:CONDA_PREFIX\python.exe"
$maxConcurrentJobs = 3
$log_path = "./logs"


# ========== 中斷處理 ==========
trap {
    Write-Host "`n[中斷偵測] 停止所有背景任務中..."
    Get-Job | Remove-Job -Force -ErrorAction SilentlyContinue
    exit
}

# Remove old log files

#if (Test-Path $log_path) {
#    Remove-Item "$log_path/*" -Force
#} else {
#    New-Item -ItemType Directory -Path $log_path | Out-Null
#}

# Helper function for submitting background jobs
function Start-TrainTestJob {
    param(
        [string]$desc,
        [string]$trainCmd,
        [string]$testCmd,
        [string]$workingDir
    )

    Start-Job -ScriptBlock {
        param($desc, $trainCmd, $testCmd, $workingDir)
#        Write-Host "working dir: $workingDir"
        Set-Location $workingDir
        Write-Host "start task: $desc"
        Invoke-Expression $trainCmd
        Invoke-Expression $testCmd
        Write-Host "complete task: $desc"
    } -ArgumentList $desc, $trainCmd, $testCmd, $workingDir
}

function Wait-For-Slot {
    while ((Get-Job | Where-Object { $_.State -eq "Running" }).Count -ge $maxConcurrentJobs) {
        Start-Sleep -Seconds 1
    }
}

# ===================================================================================

# 用來記錄所有 job（可選）
$allJobs = @()

Write-Host "== epochs: 20 / 40 / 80 =="
foreach ($epoch in 20, 40, 80) {
    $workingDir = $PWD
    $bs = 32
    $lr = 0.01
    $suffix = "e${epoch}_bs${bs}_lr${lr}"
    $log_file = "${log_path}/result_${suffix}.jsonl"
    $weight_file = "weight_${suffix}.pth"
    $train_log = "${log_path}/train_log_${suffix}.log"
    $test_log  = "${log_path}/test_log_${suffix}.log"
    $trainCmd = "python ./train.py --epochs $epoch --batch-size $bs --learning-rate $lr --weight-file $weight_file --img-desc $suffix > `"$train_log`" 2>&1"
    $testCmd = "python ./test.py --weight-file $weight_file --log $log_file > `"$test_log`" 2>&1"

    Wait-For-Slot

    $job = Start-TrainTestJob "epoch=$epoch" $trainCmd $testCmd $workingDir
    $allJobs += $job
    Write-Host "add job: $job, $suffix"
}
Write-Host "==============================================================="
Write-Host ""
Write-Host "== batch size: 8 / 16 / 32 =="
foreach ($bs in 8, 16, 32) {
    $workingDir = $PWD
    $epoch = 20
    $lr = 0.01
    $suffix = "e${epoch}_bs${bs}_lr${lr}"
    $log_file = "${log_path}/result_${suffix}.jsonl"
    $train_log = "${log_path}/train_log_${suffix}.log"
    $test_log  = "${log_path}/test_log_${suffix}.log"
    $weight_file = "weight_${suffix}.pth"
    $trainCmd = "python ./train.py --epochs $epoch --batch-size $bs --learning-rate $lr --weight-file $weight_file --img-desc $suffix > `"$train_log`" 2>&1"
    $testCmd = "python ./test.py --weight-file $weight_file --log $log_file > `"$test_log`" 2>&1"

    Wait-For-Slot

    $job = Start-TrainTestJob "bs=$bs" $trainCmd $testCmd $workingDir
    $allJobs += $job
    Write-Host "add job: $job, $suffix"
}

Write-Host "==============================================================="
Write-Host ""
Write-Host "== learning rate: 0.1 / 0.01 / 0.001 =="
foreach ($lr in 0.1, 0.01, 0.001) {
    $workingDir = $PWD
    $epoch = 20
    $bs = 32
    $suffix = "e${epoch}_bs${bs}_lr${lr}"
    $log_file = "${log_path}/result_${suffix}.jsonl"
    $train_log = "${log_path}/train_log_${suffix}.log"
    $test_log  = "${log_path}/test_log_${suffix}.log"
    $weight_file = "weight_${suffix}.pth"
    $trainCmd = "python ./train.py --epochs $epoch --batch-size $bs --learning-rate $lr --weight-file $weight_file --img-desc $suffix > `"$train_log`" 2>&1"
    $testCmd = "python ./test.py --weight-file $weight_file --log $log_file > `"$test_log`" 2>&1"

    Wait-For-Slot

    $job = Start-TrainTestJob "lr=$lr" $trainCmd $testCmd $workingDir
    $allJobs += $job
    Write-Host "add job: $job, $suffix"
}
Write-Host "==============================================================="
Write-Host ""
Write-Host "== loss: cross_entropy / label_smoothing_loss / focal_loss =="
foreach ($loss in "cross_entropy", "label_smoothing_loss", "focal_loss") {
    $workingDir = $PWD
    $epoch = 20
    $bs = 32
    $lr = 0.01
    $suffix = "loss_${loss}"
    $log_file = "${log_path}/result_${suffix}.jsonl"
    $train_log = "${log_path}/train_log_${suffix}.log"
    $test_log  = "${log_path}/test_log_${suffix}.log"
    $weight_file = "weight_${suffix}.pth"
    $trainCmd = "python ./train.py --loss $loss --epochs $epoch --batch-size $bs --learning-rate $lr --weight-file $weight_file --img-desc $suffix > `"$train_log`" 2>&1"
    $testCmd = "python ./test.py --weight-file $weight_file --log $log_file > `"$test_log`" 2>&1"

    Wait-For-Slot

    $job = Start-TrainTestJob "loss=$loss" $trainCmd $testCmd $workingDir
    $allJobs += $job
    Write-Host "add job: $job, $suffix"
}

Write-Host "==============================================================="
Write-Host ""
# == 最後一次性等待所有 job ==
Write-Host "`nWaiting for all tasks to complete..."
$allJobs | Wait-Job

# （可選）顯示結果
$allJobs | ForEach-Object {
    Write-Host "`n[Job $($_.Id) 輸出]:"
    Receive-Job -Id $_.Id
}

$allJobs | Remove-Job

Write-Host ""
$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "Finish all task, total exe time: $($duration.Hours)h $($duration.Minutes)min $($duration.Seconds)sec"