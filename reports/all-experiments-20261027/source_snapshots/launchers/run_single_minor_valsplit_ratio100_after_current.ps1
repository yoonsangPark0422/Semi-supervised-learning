$ErrorActionPreference = 'Stop'
$RuntimePython = 'C:\Users\FORYOUCOM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$DualRoot = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\dual'
$WaitPid = 1296
$ResultRoot = Join-Path 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main' 'results\valsplit_ratio100_single_minor_e50_s512_seed1_2_3'
$LogDir = Join-Path $ResultRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RunnerLog = Join-Path $LogDir 'runner.log'
function Log($m) { Add-Content -LiteralPath $RunnerLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" }
function Get-PrecisionFromRecallF1([double]$Recall, [double]$F1) { $r=$Recall/100.0; $f=$F1/100.0; $den=2.0*$r-$f; if ($den -le 0) { return 0.0 }; return ($f*$r/$den)*100.0 }
function GroupMetrics($Rows, [int]$MinClass, [int]$MaxClass) { $sel=@($Rows | Where-Object { [int]$_.class -ge $MinClass -and [int]$_.class -le $MaxClass }); $rec=($sel | Measure-Object -Property accuracy -Average).Average; $f1=($sel | Measure-Object -Property f1 -Average).Average; $precVals=@($sel | ForEach-Object { Get-PrecisionFromRecallF1 ([double]$_.accuracy) ([double]$_.f1) }); $prec=($precVals | Measure-Object -Average).Average; return @{Recall=$rec; Precision=$prec; F1=$f1} }
function AddSummary([int]$Seed, $Out) {
  $runSummary = Join-Path $ResultRoot 'run_summary.csv'
  if (-not (Test-Path -LiteralPath $runSummary)) { 'method,ratio,seed,status,last_epoch,acc,macro_f1,minority_5_9_recall,minority_5_9_precision,minority_5_9_f1,tail_7_9_recall,tail_7_9_precision,tail_7_9_f1,out' | Set-Content -LiteralPath $runSummary }
  $metrics = Join-Path $Out 'train_metrics.csv'
  $perClass = Join-Path $Out 'per_class_eval.csv'
  $mrows = @(Import-Csv -LiteralPath $metrics)
  $last = $mrows[-1]
  $epoch = [int]$last.epoch
  $acc = [double]$last.test_acc_final
  $macro = [double]$last.macro_f1_final
  $pcRows = @(Import-Csv -LiteralPath $perClass | Where-Object { [int]$_.epoch -eq $epoch -and $_.model -eq 'minor' })
  $minor = GroupMetrics $pcRows 5 9
  $tail = GroupMetrics $pcRows 7 9
  $rel = $Out.Replace('C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main' + '\', '').Replace('\','/')
  ('single_minor,100,{0},done,{1},{2:F4},{3:F4},{4:F4},{5:F4},{6:F4},{7:F4},{8:F4},{9:F4},"{10}"' -f $Seed,$epoch,$acc,$macro,$minor.Recall,$minor.Precision,$minor.F1,$tail.Recall,$tail.Precision,$tail.F1,$rel) | Add-Content -LiteralPath $runSummary
}
function UpdateAggregate() {
  $runSummary = Join-Path $ResultRoot 'run_summary.csv'
  $agg = Join-Path $ResultRoot 'aggregate_summary.csv'
  $rows = @(Import-Csv -LiteralPath $runSummary | Where-Object { $_.status -eq 'done' })
  $lines = @('method,ratio,n,acc_mean,acc_std,macro_f1_mean,macro_f1_std,minority_f1_mean,minority_f1_std,tail_f1_mean,tail_f1_std')
  foreach ($g in ($rows | Group-Object method,ratio)) {
    $items=@($g.Group); $method=$items[0].method; $ratio=$items[0].ratio; $n=$items.Count
    function Mean([double[]]$v) { if ($v.Count -eq 0) { return 0 }; return ($v | Measure-Object -Average).Average }
    function Std([double[]]$v) { if ($v.Count -le 1) { return 0 }; $m=Mean $v; return [math]::Sqrt((($v | ForEach-Object { ($_-$m)*($_-$m) }) | Measure-Object -Sum).Sum / ($v.Count-1)) }
    $acc=[double[]]@($items | ForEach-Object { [double]$_.acc }); $macro=[double[]]@($items | ForEach-Object { [double]$_.macro_f1 }); $minor=[double[]]@($items | ForEach-Object { [double]$_.minority_5_9_f1 }); $tail=[double[]]@($items | ForEach-Object { [double]$_.tail_7_9_f1 })
    $lines += ('{0},{1},{2},{3:F4},{4:F4},{5:F4},{6:F4},{7:F4},{8:F4},{9:F4},{10:F4}' -f $method,$ratio,$n,(Mean $acc),(Std $acc),(Mean $macro),(Std $macro),(Mean $minor),(Std $minor),(Mean $tail),(Std $tail))
  }
  $lines | Set-Content -LiteralPath $agg
}
Log "waiting for validation comparison runner pid=$WaitPid"
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 120 }
Log 'starting single minor validation split ratio100'
foreach ($seed in @(1,2,3)) {
  $out = Join-Path $ResultRoot ("seed_$seed")
  $metrics = Join-Path $out 'train_metrics.csv'
  if (Test-Path -LiteralPath $metrics) { $rows=@(Import-Csv -LiteralPath $metrics); if ($rows.Count -gt 0 -and [int]$rows[-1].epoch -ge 50) { Log "skip complete seed=$seed"; AddSummary $seed $out; UpdateAggregate; continue } }
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  $stdout = Join-Path $LogDir ("single_minor_seed$seed.stdout.log")
  $stderr = Join-Path $LogDir ("single_minor_seed$seed.stderr.log")
  Log "start single_minor seed=$seed"
  $args=@('train_single_minor.py','--dataset','cifar10','--num-labeled','4000','--arch','wideresnet','--batch-size','32','--mu','7','--lr','0.03','--expand-labels','--imb-ratio','100','--total-steps','25600','--eval-step','512','--val-samples-per-class','500','--seed',"$seed",'--out',$out,'--num-workers','0','--no-progress','--no-save-checkpoint')
  $proc=Start-Process -FilePath $RuntimePython -ArgumentList $args -WorkingDirectory $DualRoot -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  if ($proc.ExitCode -ne 0) { throw "single_minor failed seed=$seed see $stderr" }
  AddSummary $seed $out
  UpdateAggregate
  Log "done single_minor seed=$seed"
}
Log 'single minor validation split ratio100 finished'
