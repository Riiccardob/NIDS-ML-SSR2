-   -   -
SEMPRE:
- NO EMOJI
- CODICI COMMENTATI
- CODICI PARAMETRICI
-   -   -

!!! TEST sniffer e sniff_evaluation && togliere creazione latest da models

0 - check stacking ensemble per unire piu modelli in uno con algoritmi diversi

1 - SNIFFER:
    progress bar del pcap analizzato
    

2 - per ora è binary, dovremmo farlo multiclass:
    Binary (quello che usiamo):
        2 classi: BENIGN (0) vs ATTACK (1)
        Il modello risponde: "Questo traffico e un attacco? Si/No"
    Multiclass (opzionale, non lo usiamo):
        N classi: BENIGN, DoS, DDoS, PortScan, BruteForce, etc.

3 - Dashboard

4 - README

5 - TODO DA REPORT CLAUDE
* Fix #2: Sostituire undersampling con SMOTE+Tomek

6 - dato che su kaggle faccio aprtire 3 ru diverse, ognua genera i propri artifacts, potrebbe capitare che scelgano featrue diverse per ogni modelli, da verificare e cheidere a claude in caso
