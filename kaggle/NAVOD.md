# Návod: spuštění na Kaggle z vlastního účtu (v5 a v6)

Díky za pomoc! Na svém Kaggle účtu spustíš dva běhy tréninku jazykového modelu (GPT-2, 124 M parametrů).
Všechno běží v cloudu Kaggle zdarma; tvůj počítač jen posílá příkazy a stahuje výsledky.

| běh | co dělá | kvóta Kaggle GPU |
|---|---|---|
| **v5** | krátké srovnání: Smart Filtering (výběr 1,5×) proti baseline, 1200 kroků | ~3,5 h |
| **v6** | plný benchmark, jen Smart Filtering, do val loss 3.3821 (max 4768 kroků) | ~25–27 h ve 3 sezeních |

**Kvóta:** Kaggle dává 30 h GPU týdně (obnovuje se v sobotu). v5 + v6 dohromady (~29–31 h) se do jednoho
týdne jednoho účtu nemusí vejít. Buď pusť v5 a v6 na dvou různých účtech, nebo nejdřív v5 a v6 až s čistou
kvótou (týden předtím nic jiného na GPU). Zbývající kvóta: https://www.kaggle.com/settings → **Quotas**.

## 1. Příprava (jednou)

1. **Účet na kaggle.com s ověřeným telefonem** (Settings → Phone Verification). Bez toho Kaggle nedá GPU
   ani internet a běh potichu poběží na CPU.
2. **API token:** https://www.kaggle.com/settings → **API** → **Create New Token**. Token ulož:
   ```bash
   mkdir -p ~/.kaggle && echo TVUJ_TOKEN > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   ```
   Token nikomu neposílej a nedávej do gitu.
3. **Nástroje** (Python 3.10+, git; na Windows použij WSL nebo Git Bash):
   ```bash
   pip install kaggle
   git clone https://github.com/vbtronic/BottleCapAI_NoCap-Test.git
   cd BottleCapAI_NoCap-Test
   kaggle config view          # musí ukázat tvoje uživatelské jméno
   ```
4. **Přepiš uživatelské jméno** v obou nastaveních (místo `TVOJE_JMENO` dej svoje jméno z Kaggle):
   ```bash
   sed -i.bak 's#viktorbrunclk/#TVOJE_JMENO/#' kaggle/kernel-metadata.json kaggle/full/kernel-metadata.json
   ```

## 2. v5 — krátké srovnání (~3,5 h)

```bash
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
kaggle kernels status TVOJE_JMENO/nocap-smartfilter      # QUEUED → RUNNING → COMPLETE
```

- `--accelerator NvidiaTeslaT4` je nutné (2× T4), jinak Kaggle přidělí jiné nebo žádné GPU.
- Když běh visí ve **QUEUED / „Starting…“ déle než ~30 min**, zruš ho na webu (notebook → Versions → ⋯ →
  Cancel) a pošli znovu. Čekání ve frontě kvótu nebere.
- Po stavu **COMPLETE** stáhni výsledky a vyhodnoť:
  ```bash
  kaggle kernels output TVOJE_JMENO/nocap-smartfilter -p results/kaggle-v5
  python experiments.py eval results/kaggle-v5/quick
  ```
- Pošli Viktorovi složku `results/kaggle-v5` (je malá, jen logy a graf) a tabulku z posledního příkazu.

## 3. v6 — plný benchmark (3 sezení, ~25–27 h)

1. **Výběr:** v `kaggle/full/full_run.py` je řádek `POOL = 1.25`. Viktor ti řekne, jestli ho podle v5 změnit
   na `1.5`. Měnit jen před prvním sezením, nikdy uprostřed běhu.
2. **První sezení:**
   ```bash
   kaggle kernels push -p kaggle/full --accelerator NvidiaTeslaT4
   kaggle kernels status TVOJE_JMENO/nocap-full
   ```
   Sezení samo skončí po ~11,5 h (před 12h limitem Kaggle) a uloží checkpoint.
3. **Každé další sezení** (když je stav **COMPLETE**):
   ```bash
   ./kaggle/full/next_session.sh
   ```
   Skript stáhne logy do `results/kaggle-full/`, nahraje checkpoint (~1,5 GB) jako tvůj **soukromý** dataset
   `nocap-ckpt` a spustí další sezení, které na něj naváže. Stahuje a nahrává 1,5 GB, takže chvíli trvá.
4. Opakuj krok 3, dokud skript nenapíše **„the run is finished“** (obvykle po 3. sezení). Pak:
   ```bash
   python experiments.py eval-full results/kaggle-full
   ```
   a pošli Viktorovi výstup a složku `results/kaggle-full` (bez souborů `.pt`, ty skript maže).

## Když něco nejde

| problém | řešení |
|---|---|
| `nvidia-smi: not found` v logu | účet nemá ověřený telefon, nebo chybí `--accelerator NvidiaTeslaT4` |
| `401` / `Unauthorized` | špatný nebo neuložený token (krok 1.2) |
| `Permission denied` u skriptu | `chmod +x kaggle/full/next_session.sh` |
| běh spadl (`ERROR`) | `kaggle kernels output … -p chyba` a pošli Viktorovi obsah složky `chyba` |
| došla kvóta | počkej na sobotu, další sezení v6 naváže na poslední checkpoint |
