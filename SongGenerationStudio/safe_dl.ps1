# safe_dl.ps1 - Resilient downloading of model.pt with resume + retry loop
$ErrorActionPreference = 'Continue'
$url = 'https://huggingface.co/lglg666/SongGeneration-base/resolve/main/model.pt'
$out = 'H:\Projetos\Coding\SongGeneration-Studio\app\songgeneration_base\model.pt'
$target = 11318365872
$log = 'H:\Projetos\Coding\SongGeneration-Studio\safe_dl.log'
$marker = 'H:\Projetos\Coding\SongGeneration-Studio\safe_dl.DONE'

Remove-Item $marker -ErrorAction SilentlyContinue
for ($i = 0; $i -lt 60; $i++) {
    $sz = 0
    if (Test-Path $out) { $sz = (Get-Item $out).Length }
    Add-Content $log ("{0} try={1} size={2:N1} MB" -f (Get-Date -Format 'HH:mm:ss'), $i, ($sz / 1MB))
    if ($sz -ge $target) { Add-Content $log 'SIZE_OK'; Set-Content $marker 'complete'; exit 0 }
    if ($sz -gt 0) {
        curl.exe -sSL --ssl-no-revoke -C - -o $out $url
    } else {
        curl.exe -sSL --ssl-no-revoke -o $out $url
    }
    Add-Content $log "curl exit=$LASTEXITCODE"
    Start-Sleep -Seconds 3
}
Add-Content $log 'GAVE UP AFTER 60 TRIES'