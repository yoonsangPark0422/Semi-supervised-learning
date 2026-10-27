$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$PythonExe = 'C:\Users\FORYOUCOM\Desktop\DeCon\.venv-cu128\Scripts\python.exe'
$TrainFile = Join-Path $PSScriptRoot 'train_head_tail_exchange_sampler_alpha_val_split.py'
$WorkingDir = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\dual'
$OutputDir = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\results\dual_real_last_2_self_confidence_validation_cifar10_e50_s512_seed1'
if (Test-Path -LiteralPath $OutputDir) { throw 'Output already exists; refusing to overwrite.' }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
$TrainArgs = @(
    '-B','-u',('"'+$TrainFile+'"'),
    '--dataset','cifar10','--seed','1',
    '--num-labeled','4000','--expand-labels','--imb-ratio','100',
    '--arch','wideresnet','--batch-size','32','--mu','7','--lr','0.03',
    '--total-steps','25600','--eval-step','512',
    '--dual-train-mode','dual_sampler_weighted_ce',
    '--minor-balanced-sampler-alpha','1.2',
    '--val-samples-per-class','500','--diagnose-disagreement',
    '--minor-self-confidence-weight',
    '--num-workers','2','--no-progress','--out',('"'+$OutputDir+'"')
)
$TrainArgs | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDir 'arguments.json') -Encoding UTF8
Copy-Item -LiteralPath $TrainFile -Destination (Join-Path $OutputDir 'training_source.py')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'correction_gate.py') -Destination (Join-Path $OutputDir 'correction_gate_source.py')
Add-Content -LiteralPath (Join-Path $OutputDir 'run.log') -Value ((Get-Date -Format o)+' START seed1 epochs50 fresh validation')
$Process = Start-Process -FilePath $PythonExe -ArgumentList $TrainArgs -WorkingDirectory $WorkingDir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutputDir 'stdout.log') -RedirectStandardError (Join-Path $OutputDir 'stderr.log')
$ProcessHandle = $Process.Handle
Add-Content -LiteralPath (Join-Path $OutputDir 'run.log') -Value ('PID '+$Process.Id)
$Process.WaitForExit()
if ($Process.ExitCode -ne 0) { throw "Training exit code $($Process.ExitCode)" }
$Rows = Import-Csv -LiteralPath (Join-Path $OutputDir 'train_metrics.csv')
if ($Rows[-1].epoch -ne '50') { throw 'Epoch 50 not completed' }
Add-Content -LiteralPath (Join-Path $OutputDir 'run.log') -Value ((Get-Date -Format o)+' DONE epoch50')
