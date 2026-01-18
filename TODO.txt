-   -   -
SEMPRE:
- NO EMOJI
- CODICI COMMENTATI
- CODICI PARAMETRICI
-   -   -

1 - testare nuovi codici training (ho integrato modulo timing) | mal che vada ho lasciato la vecchia versione commentata

2 - Sistemare errore sniffer
    GIA FATTO --> lo sniffer deve essere quanto più parametrico possibile: 
    --avere la possibilità di decidere se sniffare i pacchetti live o analizzare il contenuto di un file pcap
    --di base deve utilizzare il best_model (*?? check se evaluation lo crea??), però dev'essere possibile cambiare modello 
    --deve sempre fare un logging completo dove mostra quanto più possibile 
    --opzione per selezionare la modalità in cui interagisce effettivamente col firewall modificandolo (e fa anche logging) oppure opzione in cui non modifica il firewall e si limita al logging

3 - integrare modulo timing negli altri file
    GIA FATTO --> sistema di logging delle tempistiche di esecuzione che differenzi anche tra i parametri utilizzati + script per generare statistiche finali

4 - setup colab per runnare gli script di training (lentissimi e super onerosi, soprattutto se fatti al max dei parametri)

5 - Dashboard

6 - README


