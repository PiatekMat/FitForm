"""
Plan pod pipeline users

1. Importy

2. Potem wczytywanie URL do bazy z .enva (tak jak w daily_logs)

3. Potem ustawianie loggera (tak jak w daily_logs, tyle że da się inną nazwę
                              pliku, w którym zapisują się logi)

4. Ekstrakcja danych (tak jak w daily_logs)

5. Transformacja, początek tak jak w daily_logs, ale
    po mapping usuwamy wszystkie kolumny, oprócz płci
    i dodajemy do DataFrame płeć, która najczęściej
    się powtarzała w kolumnie "Płeć" wpisywanej przez
    użytkownika, a także dajemy z automatu nazwę (trzeba
    będzie znowu rozdzielić czyjeś id od pliku i dokleić
    je do usera)

6. Wczytywanie danych do bazy danych, zresztą bardzo podobnie
    jak w daily_logs. Trzeba zrobić UPSERT

7. Main - blok wykonawczy też bardzo podobnie jak w daily_logs,
           ale bez archiwizacji ALBO ZROBIENIE NOWEGO PLIKU,
           KTÓRY PO KOLEI (TAK JAK POWINNO BYĆ) WYKONUJE CAŁE
           WCZYTANIE DO BAZY DANYCH

8. __main__ = __name__ wiadomix
"""
