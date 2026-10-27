$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$PythonExe = 'C:\Users\FORYOUCOM\Desktop\DeCon\.venv-cu128\Scripts\python.exe'
$WorkingDir = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\dual'
$ResultRoot = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\results\reduced10pct_decon_real_last2_cifar10_seed1'
$Entry = Join-Path $PSScriptRoot 'reduced_data_experiment.py'
$Common = @('--dataset','cifar10','--seed','1','--num-labeled','400','--expand-labels',
    '--arch','wideresnet','--batch-size','32','--mu','7','--lr','0.03',
    '--wdecay','0.0005','--ema-decay','0.999','--total-steps','25600','--eval-step','512',
    '--val-samples-per-class','0','--num-workers','2','--no-progress','--eval-final-only')
$Runs = @(
    @{Name='real_last_dual_2'; Extra=@('--imb-ratio','100','--dual-train-mode','dual_sampler_weighted_ce',
        '--minor-balanced-sampler-alpha','1.2','--ensemble-minor-weight','0.8',
        '--major-cross-skip-minor-conf-threshold','-1','--min-threshold','0.5')},
    @{Name='decon'; Extra=@('--match-dual-data','--num-max','162','--num-max-u','500',
        '--imb-ratio-label','100','--imb-ratio-unlabel','1',
        '--tau','1.0','--ema-u','0.99','--tau3','2.0','--tau4','1.0')}
)
foreach($Run in $Runs) {
    $Run.Out=Join-Path $ResultRoot $Run.Name
    if(Test-Path -LiteralPath $Run.Out){throw "Output exists: $($Run.Out)"}
}
function Log([string]$Message) {
    $line=(Get-Date -Format o)+' '+$Message
    Add-Content -LiteralPath (Join-Path $ResultRoot 'queue.log') -Value $line
    Write-Output $line
}
function Invoke-Experiment($Run,[bool]$AuditOnly) {
    $OutDir=if($AuditOnly){Join-Path $ResultRoot ('audit_'+$Run.Name)}else{$Run.Out}
    if(Test-Path -LiteralPath $OutDir){throw "Output exists: $OutDir"}
    New-Item -ItemType Directory -Path $OutDir | Out-Null
    $TrainArgs=@('-B','-u',('"'+$Entry+'"'),'--method',$Run.Name)+$Common+$Run.Extra+@('--out',('"'+$OutDir+'"'))
    if($AuditOnly){$TrainArgs+=@('--audit-only')}
    $TrainArgs|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $OutDir 'arguments.json')
    Log "START $($Run.Name) audit_only=$AuditOnly"
    $Process=Start-Process -FilePath $PythonExe -ArgumentList $TrainArgs -WorkingDirectory $WorkingDir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $OutDir 'stdout.log') -RedirectStandardError (Join-Path $OutDir 'stderr.log')
    $ProcessHandle=$Process.Handle
    Log "PID $($Run.Name) $($Process.Id)"
    $Process.WaitForExit()
    if($Process.ExitCode -ne 0){throw "$($Run.Name) failed exit=$($Process.ExitCode)"}
    if(!(Test-Path -LiteralPath (Join-Path $OutDir 'actual_data_audit.json'))){throw 'Missing verified data audit'}
    if(!$AuditOnly){
        $Rows=@(Import-Csv -LiteralPath (Join-Path $OutDir 'per_class_eval.csv'))
        if(!$Rows.Count -or ($Rows|Where-Object {[int]$_.epoch -ne 50 -or [double]$_.support -ne 1000})){throw 'Unexpected evaluation split or epoch'}
        if(!(Test-Path -LiteralPath (Join-Path $OutDir 'checkpoint.pth.tar'))){throw 'Checkpoint missing'}
    }
    Log "DONE $($Run.Name) audit_only=$AuditOnly"
}
try {
    Copy-Item -LiteralPath $Entry -Destination (Join-Path $ResultRoot 'split_wrapper_source.py')
    foreach($Run in $Runs){Invoke-Experiment $Run $true}
    $A=Get-Content -LiteralPath (Join-Path $ResultRoot 'audit_real_last_dual_2\actual_data_audit.json') -Raw|ConvertFrom-Json
    $B=Get-Content -LiteralPath (Join-Path $ResultRoot 'audit_decon\actual_data_audit.json') -Raw|ConvertFrom-Json
    if($A.sha256 -ne $B.sha256){throw 'Model data manifests differ'}
    Log 'BOTH_DATA_LOADERS_VERIFIED_IDENTICAL'
    foreach($Run in $Runs){Invoke-Experiment $Run $false}
    Log 'ALL_DONE'
} catch {Log "FAILED $($_.Exception.Message)";throw}
