# run.ps1


# Remove old log file if it exists
$log_path = "./logs"
#if (Test-Path $log_path) {
#    Remove-Item "$log_path/*" -Force
#} else {
#    New-Item -ItemType Directory -Path $log_path
#}

#Write-Host "== epochs: 20 / 40 / 80 =="
#
#foreach ($epoch in 20, 40, 80) {
#    $bs = 32
#    $lr = 0.01
#    $suffix = "e${epoch}_bs${bs}_lr${lr}"
#    $log_file = "${log_path}/test_log_${suffix}.jsonl"
#    Write-Host "Running with epochs=$epoch, batch_size=$bs, lr=$lr"
#    python train.py --epochs $epoch --batch-size $bs --learning-rate $lr
#    python test.py --weight "weight_${suffix}.pth" --log $log_file
#}
#
#Write-Host "== 調整 batch size: 8 / 16 =="
#foreach ($bs in 8, 16, 32) {
#    $epoch = 20
#    $lr = 0.01
#    $suffix = "e${epoch}_bs${bs}_lr${lr}"
#    $log_file = "${log_path}/test_log_${suffix}.jsonl"
#    Write-Host "Running with epochs=$epoch, batch_size=$bs, lr=$lr"
#    python train.py --epochs $epoch --batch-size $bs --learning-rate $lr
#    python test.py --weight "weight_${suffix}.pth" --log $log_file
#}
#
#Write-Host "== 調整 learning rate: 0.1 / 0.001 =="
#
#foreach ($lr in 0.1,0.01, 0.001) {
#    $epoch = 20
#    $bs = 32
#    $suffix = "e${epoch}_bs${bs}_lr${lr}"
#    $log_file = "${log_path}/test_log_${suffix}.jsonl"
#    Write-Host "Running with epochs=$epoch, batch_size=$bs, lr=$lr"
#    python train.py --epochs $epoch --batch-size $bs --learning-rate $lr
#    python test.py --weight "weight_${suffix}.pth" --log $log_file
#}

Write-Host "測試不同 loss"

foreach ($loss in "dice_loss") {
    $epoch = 2
    $bs = 32
    $lr = 0.01
    $suffix = "loss_${loss}"
    $weight_file = "weight_${suffix}.pth"
    $log_file = "${log_path}/test_log_${suffix}.jsonl"
    Write-Host "Running with loss=$loss, batch_size=32, lr=0.01"
    python train.py --loss $loss --epochs $epoch --batch-size $bs --learning-rate $lr --weight-file $weight_file
    python test.py --weight-file "weight_${suffix}.pth" --log $log_file
}
