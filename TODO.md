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

5- sistemare jupyer phase 1 tuning


6 - TODO DA REPORT CLAUDE
* Fix #2: Sostituire undersampling con SMOTE+Tomek


CHECK: dopo il cabmi in feature engineering co xgboost si ha che i mdoello xgboost ha perforamto moto meglio, ma alleanendo il lightgbmha performato peggio