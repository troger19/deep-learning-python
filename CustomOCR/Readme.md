# Projekt pre extrahovanie udajov z faktur

## Kroky

1. detekcia QR kodu : zistenie, či sa v danej faktúre vyskytuje QR kód, lokalizovať stranu faktúry a extrahovať informáciu z QR kódu. Predpokladame, ze je iba 1 QR kod v celej fakture. 
2. klasifikacia vstupu : určenie, o aký typ faktúry ide (Orange, O2, ČSOB, ine..).Extrahuje sa iba strana, na ktorej sa nachádza QR kod. V prípade, ze sa QR kód nenašiel, extrahuje sa všetko. 
 - Klasifikacia moze prebiehať 2 sposobmi : 
       - OCR sa prebehne celá strana a hľadajú sa kľúčové slová vystaviteľa faktúry : Orange, O2 s.r.o. Metodova, ČSOB, ...
       - cez NN ako Image Classification úloha  
3. Podľa klasifikácie / určenia typu faktúry, sa vyberie šablóna s ROI (Regions of interest) - oblasti (segmenty na obázku dané ich umiestnením x,y,w,h), ktoré chceme prečítať cez OCR
4. Prečítanie vybraných oblastí pomocou OCR a uloženie údajov do suboru (json, csv)
5. Kontrola údajov a kompletnosti extrakcie : porovannie údanov s údajmi z QR kódu,sčítanie čiastkových súm, DPH a porovnanie s celkovou sumou. Kontrola adresy, názvu spoločnosti, formátu IBAN, ...


 

