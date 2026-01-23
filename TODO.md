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

5- check fomrula score cone le foto su WA

6- check scelta parametri random forest: magari creare un modulo per il testing dei parametri ottimale che random_forest.py richiama in base ad un flag




CALDUE
il punto 1 è stato compeltato creando il file src/hyperparameter_tuning.py

per quanto riguarda il punto 2 e il punto 3 (rispettivamente le modifiche agli algoritmi di training (RF; xgboost, lightbm) ho integrato le modifiche che mi hai consigliato nel random forest e hanno avuto successo. Vorrei che mi mandassi le rispettive modifiche da dover fare ai codici di lightbm e xgboost secondo la logica di quelle applicate a random forest.


Per quanto riguarda il model versioning che conteine la fuzione usata per salvare i modelli dopo queste modifiche non funziona più in modo ottimale. Salva i mdoelli in cartelle della forma cvX_iterY. Il problema è che attualmente con queste modifiche prima si esegue hyperparameter_tuning.py e poi con questi parametri "ottimali e uguali per tutti gli allenamenti" si va ad allenare i modelli.

Mi chiedevo dunque se fosse necessario in primis avere ancora questo modulo in quanto il suo utilizzo principale era per salvare le varie versioni dei parametri quandoa ncora venivano calcoalti durante il training e ora per come è strutturato tutti i modelli dello stesso algoritmo verranno alleanti con gli stessi iperparametri i quali contengono sia iterations che cv 