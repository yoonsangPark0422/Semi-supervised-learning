$ErrorActionPreference = 'Stop'
$RuntimePython = 'C:\Users\FORYOUCOM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$RepoRoot = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main'
$DualRoot = Join-Path $RepoRoot 'dual'
$DeConRoot = 'C:\Users\FORYOUCOM\Desktop\DeCon'
$ResultRoot = Join-Path $RepoRoot 'results\official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3'
$LogDir = Join-Path $ResultRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RunnerLog = Join-Path $LogDir 'runner.log'
function Log($m) { Add-Content -LiteralPath $RunnerLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" }
function Mean([double[]]$v) { if ($v.Count -eq 0) { return 0 }; return ($v | Measure-Object -Average).Average }
function Std([double[]]$v) { if ($v.Count -le 1) { return 0 }; $m=Mean $v; return [math]::Sqrt((($v | ForEach-Object { ($_-$m)*($_-$m) }) | Measure-Object -Sum).Sum / ($v.Count-1)) }
function Get-PrecisionFromRecallF1([double]$Recall, [double]$F1) { $r=$Recall/100.0; $f=$F1/100.0; $den=2.0*$r-$f; if ($den -le 0) { return 0.0 }; return ($f*$r/$den)*100.0 }
function GroupMetrics($Rows, [int]$MinClass, [int]$MaxClass) {
  $sel=@($Rows | Where-Object { [int]$_.class -ge $MinClass -and [int]$_.class -le $MaxClass })
  if ($sel.Count -eq 0) { return @{Recall=0; Precision=0; F1=0} }
  $rec=($sel | Measure-Object -Property accuracy -Average).Average
  $f1=($sel | Measure-Object -Property f1 -Average).Average
  $precVals=@($sel | ForEach-Object { Get-PrecisionFromRecallF1 ([double]$_.accuracy) ([double]$_.f1) })
  $prec=($precVals | Measure-Object -Average).Average
  return @{Recall=$rec; Precision=$prec; F1=$f1}
}
function UpdateAggregate() {
  $runSummary = Join-Path $ResultRoot 'run_summary.csv'
  $agg = Join-Path $ResultRoot 'aggregate_summary.csv'
  if (-not (Test-Path -LiteralPath $runSummary)) { return }
  $rows = @(Import-Csv -LiteralPath $runSummary | Where-Object { $_.status -eq 'done' })
  $lines = @('dataset,method,ratio,n,acc_mean,acc_std,macro_f1_mean,macro_f1_std,minority_5_9_f1_mean,minority_5_9_f1_std,tail_7_9_f1_mean,tail_7_9_f1_std')
  foreach ($g in ($rows | Group-Object dataset,method,ratio)) {
    $items=@($g.Group); $dataset=$items[0].dataset; $method=$items[0].method; $ratio=$items[0].ratio; $n=$items.Count
    $acc=[double[]]@($items | ForEach-Object { [double]$_.acc }); $macro=[double[]]@($items | ForEach-Object { [double]$_.macro_f1 }); $minor=[double[]]@($items | ForEach-Object { [double]$_.minority_5_9_f1 }); $tail=[double[]]@($items | ForEach-Object { [double]$_.tail_7_9_f1 })
    $lines += ('{0},{1},{2},{3},{4:F4},{5:F4},{6:F4},{7:F4},{8:F4},{9:F4},{10:F4},{11:F4}' -f $dataset,$method,$ratio,$n,(Mean $acc),(Std $acc),(Mean $macro),(Std $macro),(Mean $minor),(Std $minor),(Mean $tail),(Std $tail))
  }
  $lines | Set-Content -LiteralPath $agg
}
function AddSummary($Dataset, $Method, [int]$Seed, $Out, $ModelName, $AccField, $MacroField) {
  $runSummary = Join-Path $ResultRoot 'run_summary.csv'
  if (-not (Test-Path -LiteralPath $runSummary)) { 'dataset,method,ratio,seed,status,last_epoch,acc,macro_f1,minority_5_9_recall,minority_5_9_precision,minority_5_9_f1,tail_7_9_recall,tail_7_9_precision,tail_7_9_f1,out' | Set-Content -LiteralPath $runSummary }
  $metrics = Join-Path $Out 'train_metrics.csv'
  $perClass = Join-Path $Out 'per_class_eval.csv'
  if (Test-Path -LiteralPath $metrics) {
    $mrows = @(Import-Csv -LiteralPath $metrics)
    $last = $mrows[-1]
    $epoch = [int]$last.epoch
    $acc = [double]$last.$AccField
    $macro = [double]$last.$MacroField
  } else {
    $pcAll = @(Import-Csv -LiteralPath $perClass | Where-Object { $_.model -eq $ModelName })
    $epoch = ($pcAll | ForEach-Object { [int]$_.epoch } | Measure-Object -Maximum).Maximum
    $pcLastForMacro = @($pcAll | Where-Object { [int]$_.epoch -eq $epoch })
    $macro = ($pcLastForMacro | Measure-Object -Property f1 -Average).Average
    $logPath = Join-Path $Out 'log.txt'
    $acc = 0.0
    if (Test-Path -LiteralPath $logPath) {
      Get-Content -LiteralPath $logPath | ForEach-Object {
        if ($_ -match '^\s*([0-9]+\.?[0-9]*)\s+') { $script:lastLogAcc = [double]$matches[1] }
      }
      $acc = [double]$script:lastLogAcc
    } else {
      $acc = ($pcLastForMacro | Measure-Object -Property accuracy -Average).Average
    }
  }
  $pcRows = @(Import-Csv -LiteralPath $perClass | Where-Object { [int]$_.epoch -eq $epoch -and $_.model -eq $ModelName })
  $minor = GroupMetrics $pcRows 5 9
  $tail = GroupMetrics $pcRows 7 9
  $rel = $Out.Replace($RepoRoot + '\', '').Replace('\','/')
  ('{0},{1},100,{2},done,{3},{4:F4},{5:F4},{6:F4},{7:F4},{8:F4},{9:F4},{10:F4},{11:F4},"{12}"' -f $Dataset,$Method,$Seed,$epoch,$acc,$macro,$minor.Recall,$minor.Precision,$minor.F1,$tail.Recall,$tail.Precision,$tail.F1,$rel) | Add-Content -LiteralPath $runSummary
  UpdateAggregate
}
function SummaryExists($Dataset, $Method, [int]$Seed) {
  $runSummary = Join-Path $ResultRoot 'run_summary.csv'
  if (-not (Test-Path -LiteralPath $runSummary)) { return $false }
  $rows = @(Import-Csv -LiteralPath $runSummary | Where-Object { $_.dataset -eq $Dataset -and $_.method -eq $Method -and [int]$_.seed -eq $Seed -and $_.status -eq 'done' })
  return $rows.Count -gt 0
}
function Ensure-Run($Dataset, $Method, [int]$Seed, [string]$WorkDir, [string[]]$ArgsList, $Out, $ModelName, $AccField, $MacroField) {
  if (SummaryExists $Dataset $Method $Seed) {
    Log "skip summarized dataset=$Dataset method=$Method seed=$Seed"
    return
  }
  $perClass = Join-Path $Out 'per_class_eval.csv'
  $metrics = Join-Path $Out 'train_metrics.csv'
  $logPath = Join-Path $Out 'log.txt'
  if ((Test-Path -LiteralPath $perClass) -and ((Test-Path -LiteralPath $metrics) -or (Test-Path -LiteralPath $logPath))) {
    Log "summarize existing dataset=$Dataset method=$Method seed=$Seed"
    AddSummary $Dataset $Method $Seed $Out $ModelName $AccField $MacroField
    return
  }
  Invoke-Run $Dataset $Method $Seed $WorkDir $ArgsList $Out
  AddSummary $Dataset $Method $Seed $Out $ModelName $AccField $MacroField
}
function Invoke-Run($Dataset, $Method, [int]$Seed, [string]$WorkDir, [string[]]$ArgsList, $Out) {
  New-Item -ItemType Directory -Force -Path $Out | Out-Null
  $stdout = Join-Path $LogDir ("$Dataset`_$Method`_seed$Seed.stdout.log")
  $stderr = Join-Path $LogDir ("$Dataset`_$Method`_seed$Seed.stderr.log")
  Log "start dataset=$Dataset method=$Method seed=$Seed out=$Out"
  $proc = Start-Process -FilePath $RuntimePython -ArgumentList $ArgsList -WorkingDirectory $WorkDir -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  if ($proc.ExitCode -ne 0) { throw "$Dataset $Method seed=$Seed failed with exit=$($proc.ExitCode); see $stderr" }
  Log "done dataset=$Dataset method=$Method seed=$Seed"
}
$Seeds = @(1,2,3)
$Datasets = @('cifar10','svhn')
Log 'official test 3-model run started'
foreach ($dataset in $Datasets) {
  foreach ($seed in $Seeds) {
    if ($dataset -eq 'cifar10') {
      $batch = '32'; $dualScript='train_head_tail_exchange_sampler_alpha_val_split.py'; $minorScript='train_single_minor_sampler_alpha.py'; $dualExtra=@('--expand-labels','--ensemble-minor-weight','0.8','--minor-balanced-sampler-alpha','1.2','--major-cross-skip-minor-conf-threshold','-1'); $minorExtra=@('--expand-labels','--minor-balanced-sampler-alpha','1.2');
    } else {
      $batch = '64'; $dualScript='train_head_tail_exchange_sampler_alpha_val_split.py'; $minorScript='train_single_minor_sampler_alpha.py'; $dualExtra=@('--ensemble-minor-weight','0.7','--minor-balanced-sampler-alpha','1.2','--major-cross-skip-minor-conf-threshold','-1'); $minorExtra=@('--minor-balanced-sampler-alpha','1.2')
    }
    $out = Join-Path $ResultRoot "$dataset\dual\seed_$seed"
    $args = @($dualScript,'--dataset',$dataset,'--num-labeled','4000','--arch','wideresnet','--batch-size',$batch,'--mu','7','--lr','0.03','--imb-ratio','100','--total-steps','25600','--eval-step','512','--dual-train-mode','dual_sampler_weighted_ce','--val-samples-per-class','0','--seed',"$seed",'--out',$out,'--num-workers','0','--no-progress','--no-save-checkpoint','--eval-final-only') + $dualExtra
    Ensure-Run $dataset 'dual' $seed $DualRoot $args $out 'ensemble' 'test_acc_final' 'macro_f1_final'

    $out = Join-Path $ResultRoot "$dataset\minor\seed_$seed"
    $args = @($minorScript,'--dataset',$dataset,'--num-labeled','4000','--arch','wideresnet','--batch-size',$batch,'--mu','7','--lr','0.03','--imb-ratio','100','--total-steps','25600','--eval-step','512','--val-samples-per-class','0','--seed',"$seed",'--out',$out,'--num-workers','0','--no-progress','--no-save-checkpoint','--eval-final-only') + $minorExtra
    Ensure-Run $dataset 'minor' $seed $DualRoot $args $out 'minor' 'test_acc_final' 'macro_f1_final'

    $out = Join-Path $ResultRoot "$dataset\decon\seed_$seed"
    $args = @('train_val_split.py','--dataset',$dataset,'--num-labeled','4000','--num-max','1614','--num-max-u','4000','--imb-ratio-label','100','--imb-ratio-unlabel','100','--arch','wideresnet','--batch-size',$batch,'--mu','7','--lr','0.03','--total-steps','25600','--eval-step','512','--tau','1.0','--ema-u','0.99','--tau3','2.0','--tau4','1.0','--val-samples-per-class','0','--seed',"$seed",'--out',$out,'--num-workers','0','--no-progress','--no-save-checkpoint','--eval-final-only')
    Ensure-Run $dataset 'decon' $seed $DeConRoot $args $out 'decon' 'test_acc' 'macro_f1'
  }
}
Log 'official test 3-model run finished'