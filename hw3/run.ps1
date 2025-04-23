$datasets = @("data_1.txt", "data_2.txt", "data_3.txt")
Remove-Item -Path ./*.png
Remove-Item -Path ./*.txt
Pause

foreach ($dataset in $datasets) {
    echo "Run Linear Regression on $dataset"
    python Linear_Regression.py --path ./dataset/$dataset | Out-File ./Result_${dataset}_result.txt
    echo "Generate Linear Regression Plot"
    Rename-Item -Path ./Linear_Regression.png -NewName LinearRegression_${dataset}.png
}