# CarePattern

## Introductie

CarePattern is een prototypeproject binnen CARAI gericht op het verminderen van administratieve lasten in de zorg. Het systeem registreert fysieke handelingen van zorgpersoneel automatisch via sensordata, videobeelden of handmatige input. Deze data worden geanalyseerd om patronen van aanwezigheid en handelingen te herkennen en te koppelen aan administratieve registraties.

## Gebruik

Door `main.py` uit te voeren wordt de Flask applicatie gestart.

In de browser kan de applicatie worden geopend op `http://localhost:8080/`.

De huidige configuratie kan worden bekeken op `http://localhost:8080/config`.

Wanneer de applicatie gebruikt word, zal er in de projectfolder een `instance` folder verschijnen.
In deze folder worden de geÜploadde en gegenereerde bestanden opgeslagen.
Deze folder kan handmatig worden verwijderd, wanneer de applicatie niet meer gebruikt wordt.
Dit moet ook gedaan worden om de cache van de applicatie te wissen.


## Configuratie
Bij het starten van de applicatie wordt de `config.ini` file gelezen. Dit inladen gebeurt in de `config_loader.py` file van de frontend.
In dit bestand kunnen verschillende configuratie opties worden ingesteld.

## Aanpassen

Voor toekomstige aanpassingen of hergebruik kan de volgende structuur worden aangehouden.
De `carepattern` module is opgesplitst in 2 subpackages; `core` en `frontend`:

- In `carepattern.core` bevindt zich de core functionaliteit van CarePattern.
- In `carepattern.frontend` bevindt zich een frontend interface voor de functionaliteit uit de `core` package.

### `carepattern.core`

In de `core` package bevinden zich de volgende modules:
- `detect.py`: Hierin zit alle logica voor het detecteren van patronen, op basis van de gedetecteerde data.
- `jobs.py`: Hierin zit alle logica voor het verwerken van jobs. Deze worden gebruikt om asynchrone verwerking van data te realiseren.
- `video.py`: Hierin zit alle logica voor het verwerken van videobeelden.

### `carepattern.frontend`

In de `templates` folder kunnen HTML bestanden worden geplaatst die ingelezen kunnen worden door de Flask applicatie.
In deze HTML bestanden kunnen Jinja2 templates worden gebruikt om dynamische content weer te geven.

Ook bevat de router van de frontend in `__init__.py` de logica voor het inladen en verwerken van bestanden.
