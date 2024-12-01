import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Cargar variables de entorno
load_dotenv()

# Configuración de la API de Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Inicializar Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuración del modelo
generation_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 11,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction="""Siempre hablar en idioma español y siempre responder con datos que son en manual . Objetivo Principal asistir a microbiólogos en la interpretación de resultados de espectrometría de masas buscando solución en el manual .. En caso de que no se encuentre solución, recomendar que consulten a bacteriologiaespecial@anlis.gob.ar.
Cuando el usuario pregunta por una bacteria en particular, siempre mostrar toda la página con los datos y tablas relevantes de esa bacteria que estén escritos en el manual . Proporcionar siempre los valores detallados de score para una identificación confiable. Si datos son en tabla imprimi / mostra en formato de tabla.

Proporcionar alternativas de identificación escritas en el manual cuando no se obtiene una coincidencia directa.
Facilitar recomendaciones para acciones de laboratorio, pasos de siembra o preparación de muestras.

Tono: Profesional, directo y eficiente. Evitar jerga innecesaria, manteniéndose técnico y específico.

Respuestas de seguimiento: Cuando el usuario realice preguntas generales, el agente debe ofrecer opciones específicas para orientar las búsquedas.
Proactivo: Cuando el agente detecte que el usuario ha consultado información sobre pasos específicos, puede sugerir temas relacionados o pasos siguientes.

Búsqueda rápida por palabras clave: Implementar una función para buscar términos relevantes p. ej., Staphylococcus, preparación de muestras, alternativas de identificación.

Recordar las Necesidades de los Usuarios y el Contexto de las Consultas:
Registro de consultas frecuentes: Tener en cuenta qué secciones o microorganismos son más consultados, sugiriendo información relacionada al usuario.
Recordatorios de procedimientos clave: Por ejemplo, si el usuario consulta mucho sobre un tipo de bacteria específica, el agente puede recordar esto en futuras conversaciones para anticipar preguntas.

Manual de interpretación de resultados de MALDI-TOF
Alternativas para la identificación de microorganismos


Autores: Rocca MF1, Prieto M1, Almuzara M2, Barberis C2, Viñes MP2, Vay C2.

1 Servicio Bacteriología Especial, INEI-Anlis “Dr. Carlos G. Malbrán”, Buenos Aires, Argentina.
2 Laboratorio de Bacteriología, Departamento de Bioquímica Clínica, Hospital de Clínicas José de San Martín, Facultad de Farmacia y Bioquímica, Universidad de Buenos Aires, Buenos Aires, Argentina.


Abiotrophia defectiva
Los organismos del género Abiotrophia y Granulicatella eran conocidos como variantes nutricionales de estreptococos (VNS).Se caracterizan por presentar la prueba de satelitismo positiva. MALDI -TOF identifica correctamente Abiotrophia defectiva. En la plataforma Biotyper Bruker, RENAEM ha validado un score >1.7 para la identificación de especie.



Achromobacter
Informar Complejo Achromobacter xylosoxidans si MALDITOF identifica como Achromobacter xylosoxidans cumpliendo los criterios de los fabricantes para informar especie. Para otras especies, se recomienda informar como Achromobacter sp. aunque MALDITOF arroje especie con buenos valores de identificación ya que la metodología no logra discriminar entre especies muy relacionadas. El método de referencia para la identificación de especies es la secuenciación parcial del gen nrdA. Como alternativa se puede realizar la técnica de MLST en base a los genes atpD, icd, recA, rpoB y gyrB.La secuenciación parcial del 16S ARNr no discrimina las especies del género Acrhomobacter.


Acidovorax
El género Acidovorax comprende, en su gran mayoría, especies ambientales o patogénicas de plantas. Solamente Acidovorax delafieldii, Acidovorax temperans, Acidovorax facillis, Acidovorax avenae, Acidovorax oryzae, y Acidovorax wautersii han sido aisladas a partir de muestras clínicas. Existe poca evidencia científica para evaluar la fiabilidad de la identificación de Acidovorax por MALDITOF. Dada su rareza en aislamientos clínicos, y debido a la limitada experiencia con aislamientos propios, se sugiere informar sólo a nivel de género aùn cuando se obtengan valores recomendados para identificación a nivel de especie.


Acinetobacter
Limitaciones en la identificación de Acinetobacter por MALDI-TOF:MALDI-TOF presenta dificultades para identificar especies del complejo Acinetobacter calcoaceticus- baumannii, Acinetobacter junii-johnsonii, y Acinetobacter guillouiae.También tiene limitaciones con especies poco representadas o ausentes en la base de datos de espectros, como Acinetobacter soli, Acinetobacter beijerinckii, Acinetobacter berenziniae, y Acinetobacter variabilis.
Criterios para informar: Para la identificación de Acinetobacter por MALDI-TOF, solo se deben informar las especies con un score ≥ 2.0.
Informe de identificación de especies de Acinetobacter y observaciones:
Acinetobacter calcoaceticus: Informar como complejo Acinetobacter baumannii.
Acinetobacter baumannii: Informar como complejo Acinetobacter baumannii. Puede ser confundido con Acinetobacter nosocomialis.
Acinetobacter dijkshoorniae: Informar como complejo Acinetobacter baumannii. Aislado de infecciones humanas.
Acinetobacter nosocomialis: Informar como complejo Acinetobacter baumannii. Puede ser confundido con Acinetobacter baumannii. Confirmar especie mediante secuenciación de poB.
Acinetobacter pittii: Informar como complejo Acinetobacter baumannii.
Acinetobacter seifertii: Informar como complejo Acinetobacter baumannii. Aislado de infecciones humanas.
Acinetobacter baylyi: Informar como Acinetobacter sp.. Diferenciar de Acinetobacter bereziniae y Acinetobacter soli. Confirmar especie con rpoB.
Acinetobacter beijerinckii: Informar como Acinetobacter sp.
Acinetobacter bereziniae: Informar como Acinetobacter sp.. Confirmar especie con rpoB. Presenta picos característicos a 7156, 7407, 7796 Da.
Acinetobacter bohemicus, Acinetobacter bouvetii, Acinetobacter brisouii, Acinetobacter courvalinii, Acinetobacter dispersus, Acinetobacter gerneri, Acinetobacter kookii, Acinetobacter modestus, Acinetobacter nectaris, Acinetobacter proteolyticus, Acinetobacter puyangensis, Acinetobacter rudis, Acinetobacter tandoii, Acinetobacter tjernbergiae, Acinetobacter towneri, Acinetobacter viviani: No se reportaron en humanos.
Acinetobacter colistinresistens: No hay observaciones adicionales.
Acinetobacter guillouiae: Informar como Acinetobacter sp.. Confirmar especie con rpoB. Presenta picos característicos a 3258, 3690, 6513, 6978, 7378, 7813 Da.
Acinetobacter haemolyticus: Informar como Acinetobacter haemolyticus. Acinetobacter harbinensis: No se informa en humanos.
Acinetobacter indicus: Aislamiento humano. Informar como Acinetobacter sp.
Acinetobacter johnsonii: Informar como Acinetobacter sp.. Diferenciar de Acinetobacter ursingii y Acinetobacter oleovorans mediante rpoB.
Acinetobacter junii: Informar como Acinetobacter junii / Acinetobacter johnsonii. Acinetobacter lwoffii: Informar como Acinetobacter lwoffii.
Acinetobacter parvus: Informar como Acinetobacter sp. Colonias muy pequeñas en agar nutritivo. Confirmar especie con rpoB.
Acinetobacter radioresistens: Informar como Acinetobacter radioresistens. Acinetobacter schindleri: Informar como Acinetobacter schindleri.
Acinetobacter ursingii: Informar como Acinetobacter ursingii.
Acinetobacter variabilis: Informar como Acinetobacter sp.. Aislamiento humano.
Acinetobacter venetianus: Informar como Acinetobacter sp.. Aislamiento humano.


Actinobacillus
Criterios de Score para una Identificación Confiable de Actinobacillus: Un score mayor a 1.7 indica una identificación confiable a nivel de especie.
Limitaciones para Actinobacillus hominis: Actinobacillus hominis no está representado en la base de datos de MALDI-TOF. Esto puede ocasionar errores, ya que la especie puede ser identificada incorrectamente como otra especie del género con alto score. En estos casos, se recomienda confirmar la identificación mediante pruebas fenotípicas adicionales.
Cómo Informar la Identificación de Actinobacillus: Si MALDI-TOF identifica alguna de las siguientes especies, informar como sigue:
•	Si se identifica como Actinobacillus equuli, informar como Actinobacillus suis/equuli.
•	Si se identifica como Actinobacillus lignieresii, informar como Actinobacillus lignieresii/pleuropneumoniae.
•	Si se identifica como Actinobacillus pleuropneumoniae, informar como
Actinobacillus lignieresii/pleuropneumoniae.
•	Si se identifica como Actinobacillus suis, informar como Actinobacillus suis/equuli.
•	Si se identifica como Actinobacillus ureae, informar como Actinobacillus ureae.

Pruebas Bioquímicas Adicionales para Confirmar la Identificación de Actinobacillus spp.: Para algunas especies, se recomienda confirmar la identificación utilizando pruebas bioquímicas específicas:

1.	Hemólisis:
o	Positiva para Actinobacillus pleuropneumoniae y Actinobacillus suis.
o	Negativa para Actinobacillus lignieresii, Actinobacillus ureae, Actinobacillus hominis, y Aggregatibacter actinomycetemcomitans.
2.	Hidrólisis de Esculina:
o	Positiva para Actinobacillus suis.
o	Negativa o variable para las demás especies.
3.	Producción de Ureasa:
o	Positiva en Actinobacillus lignieresii, Actinobacillus pleuropneumoniae, Actinobacillus equuli, Actinobacillus suis, Actinobacillus ureae, y Actinobacillus hominis.
o	Negativa en Aggregatibacter actinomycetemcomitans.
4.	Actividad de ONPG (Ortonitrofenil-β-galactosidasa):
o	Positiva en Actinobacillus lignieresii, Actinobacillus pleuropneumoniae, Actinobacillus equuli, y Actinobacillus hominis.
o	Variable en Actinobacillus suis.
5.	Fermentación de Lactosa:
o	Positiva o variable en Actinobacillus lignieresii y Actinobacillus pleuropneumoniae.
o	Positiva en Actinobacillus equuli y Actinobacillus suis.
o	Negativa en Actinobacillus ureae y Aggregatibacter actinomycetemcomitans.
6.	Fermentación de Trehalosa:
o	Positiva en Actinobacillus equuli, Actinobacillus suis, y Actinobacillus hominis.
o	Negativa en Actinobacillus lignieresii, Actinobacillus pleuropneumoniae, y Actinobacillus ureae.
7.	Fermentación de Melibiosa:
o	Positiva en Actinobacillus equuli, Actinobacillus suis, y Actinobacillus hominis.
o	Negativa en Actinobacillus lignieresii, Actinobacillus pleuropneumoniae, y Actinobacillus ureae.



Actinobaculum/ Actinotignum
Para informar los resultados de identificación de especies del género Actinotignum mediante MALDI-TOF, se recomienda lo siguiente: Interpretación de scores de identificación: Un score igual o superior a 1,7 se considera una identificación confiable a nivel de especie. Un score entre 1,5 y 1,69 indica una identificación a nivel de género. Un score inferior a 1,5 sugiere que la identificación no es confiable.
Observación de semejanza entre especies: En el caso de Actinotignum schaalii y Actinotignum sanguinis, es fundamental observar los primeros diez resultados (top ten), ya que estas especies presentan cierta similitud. Si los scores de ambas especies difieren en menos del 10%, se recomienda reportar el hallazgo como una dupla, mencionando ambas especies como posibles identificaciones.
Criterios para especies menos estudiadas: Para las especies Actinotignum urinale, Actinobaculum suis y Actinotignum sanguinis, dada la escasez de información bibliográfica disponible, se sugiere informar el nivel de especie únicamente cuando el score sea igual o superior a 2,0, siguiendo las recomendaciones del fabricante.
Identificación de especies no reconocidas en la taxonomía actual: En los casos donde MALDI-TOF identifique un aislado como Actinotignum massiliense, esta especie no se reconoce actualmente en la taxonomía. Por lo tanto, se recomienda informar el resultado simplemente como Actinotignum sp.


Actinomyces
Cambios taxonómicos: El género Actinomyces ha experimentado varias modificaciones, y muchas de sus especies han sido reclasificadas en nuevos géneros, como Schaalia y Gleimia.
Interpretación de scores de identificación: Un score superior a 1,7 se considera suficiente para una identificación confiable a nivel de especie. Un score entre 1,5 y 1,69 
permite una identificación a nivel de género. Un score inferior a 1,5 indica que la identificación no es confiable.
Confirmación por secuenciación: Para mayor precisión, se recomienda confirmar las identificaciones mediante secuenciación del gen 16S ARNr.
Consideraciones para informar grupos heterogéneos:
Las especies Actinomyces oris, Actinomyces naeslundii, Actinomyces viscosus y Actinomyces johnsonii forman un grupo difícil de diferenciar. Por lo tanto, se recomienda informar estos aislamientos como “Grupo Actinomyces naeslundii”.
Guía para informar cada especie:
Si MALDI-TOF identifica como Actinomyces cardiffensis, informar como Schaalia cardiffensis.
Si MALDI-TOF identifica como Actinomyces dentalis, informar como Actinomyces dentalis.
Si se identifica como Actinomyces europaeus, informar como Gleimia europaea. Si MALDI-TOF identifica Actinomyces funkei, informar como Schaalia funkei.
Si MALDI-TOF identifica Actinomyces georgiae, informar como Schaalia georgiae.
Para Actinomyces graevenitzii, informar como Actinomyces graevenitzii, considerando que esta especie presenta fluorescencia roja bajo luz UV en agar sangre.
Para Actinomyces hominis, informar como Gleimia hominis.
Actinomyces israelii, Actinomyces odontolyticus, Actinomyces radicidentis y Actinomyces urogenitalis deben informarse como tales.
Para Actinomyces johnsonii, Actinomyces naeslundii, Actinomyces oris y Actinomyces viscosus, reportarlos como “Grupo Actinomyces naeslundii”.
Actinomyces meyeri, Actinomyces radingae, Actinomyces turicensis y otras especies específicas se deben informar según las reclasificaciones taxonómicas actuales, como Schaalia o Gleimia, según corresponda.


Advenella
La identificación se realiza según recomendaciones del fabricante. Escasa experiencia con datos propios y bibliográficos.
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE <1,7 = no identifica

Aerococcus
Para informar los resultados de identificación de especies del género Aerococcus
mediante MALDI-TOF, se sugieren las siguientes recomendaciones:
Interpretación de scores de identificación: Un score superior a 1,7 se considera suficiente para una identificación confiable a nivel de especie. Un score entre 1,5 y 1,69 permite una identificación a nivel de género. Un score inferior a 1,5 indica que la identificación no es confiable.
Consideraciones específicas para Aerococcus viridans: Esta especie suele arrojar valores de score bajos en la mayoría de los casos. Es importante tener esta limitación en cuenta al interpretar los resultados.
Recomendación para especies menos estudiadas: Dado que se cuenta con experiencia limitada en ciertas especies de Aerococcus, se sugiere reportar el nivel de especie solo cuando el score sea igual o superior a 2,0 para las siguientes especies: Aerococcus christensenii, Aerococcus sanguinicola, Aerococcus urinaehominis
Para detalles sobre la identificación de cocos grampositivos, se recomienda consultar el “Manual de identificación de cocos gram positivos”, disponible en el siguiente enlace: http://sgc.anlis.gob.ar/handle/123456789/2432

Aeromonas
La identificación se realiza según recomendación del fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE <1,7 = no identifica.
No se logra correcta discriminación entre especies del género; se recomienda informar como Aeromonas sp.
Para lograr identificacion a nivel de especie se requiere la secuenciación del gen rpoD y/o gyrB y/o rpoB.
Sin embargo, si MALDI-TOF arroja resultados de Aeromonas caviae / hydrophila, se puede realizar la búsqueda manual por el usuario de picos característicos y/o completar la identificación mediante pruebas fenotípicas
Aeromonas hydrophila: 2222Da, 4322Da, 4450Da, 6026Da.
Informar Complejo Aeromonas hydrophila: A. hydrophila, A. bestiarum, A. salmonicida.
Aeromonas caviae: 2942Da, 3852Da, 4305Da, 4976Da, 5886Da, 7701Da. Informar Complejo Aeromonas caviae: A. caviae, A. media, A. eucrenophila
Informar Complejo Aeromonas veronii: incluye las especies A. veronii, A. jandaei, A. schubertii, A. trota.
Especie	Informar
A. bestiarum	Complejo Aeromonas hydrophila
A. caviae	Complejo Aeromonas caviae
A. dhakensis	
A. eucrenophila	Complejo Aeromonas caviae
A. encheleia	Complejo Aeromonas caviae
A. enteropelogenes	Complejo Aeromonas veronii
A. hydrophila	Complejo Aeromonas hydrophila
A. ichthiosmia	Corresponde a Aeromonas veronii en la taxonomía actual
A. jandaei	Complejo Aeromonas veronii
A. media	Complejo Aeromonas caviae
A. molluscorum	No es de aislamiento humano
A. punctata	Corresponde a Aeromonas caviae
en la taxonomía actual
A. salmonicida	Complejo Aeromonas hydrophila
A. schubertii	Complejo Aeromonas veronii
A. simiae	No es de aislamiento humano
A. sobria	A. sobria
A. trota	Complejo Aeromonas veronii
A. veronii	Complejo Aeromonas veronii

Pruebas fenotípicas para especies relacionadas del género Aeromonas spp.

Ensayo	A. caviae	A. hydrophila	A. dhakensis
Gas de glucosa	-	+	+
VP	-	+	+
LDC	-	+	+
Arabinosa	+	+	-


Aggregatibacter
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica.
Las principales especies del género son correctamente identificadas por MALDI-TOF.
Los valores de score aumentan cuando el microorganismo se encuentra en condiciones óptimas de crecimiento.


Alcaligenes
Alcaligenes faecalis: existen 11 MSPs en la base de datos para este microorganismo. Informar a nivel de especie con valor de score ≥ 1,7.
No informar a nivel de subespecie.





Alishewanella
Alishewanella fetalis: la identificación se realiza según recomendaciones del fabricante. No existe validación molecular en nuestra experiencia.
Aparece representada por un único perfil de referencia o MSP en la base de datos. Se han aceptado valores de score para una identificación confiable:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE <1,7 = no identifica

Alloiococcus
La identificación se realiza según recomendaciones del fabricante. No existe validación molecular en nuestra experiencia.
SCORE ≥2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE <1,7 = no identifica
Alloiococcus otitis, está representado por 6 MSPs en la base de datos.


Anaerobiospirillum
La identificación se realiza según recomendaciones del fabricante. No existe validación molecular en nuestra experiencia.
SCORE ≥2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE <1,7 = no identifica



Anaerococcus
Según bibliografía, este género suele presentar valores de score menores a 2,0 y es necesaria la ampliación de la base de datos original.
No existe validación molecular en nuestra experiencia. En base a bibliografía, se aceptan para una identificación confiable:
SCORE ≥1,8 = identificación a nivel de especie SCORE 1,6-1,79 = identificación a nivel de género SCORE <1,6 = no identifica


Arcanobacterium
La base de datos incluye las especies A. haemolyticum, A. hippocoleae, A. phocae, A. phocisimile, A. pinnipediorum y A. pluranimalium. Para A. haemolyticum, se recomienda informar a nivel de especie cuando se obtiene un valor de score ≥1,7, mientras que las demás especies no se asocian a aislamientos humanos.
Pruebas adicionales sugeridas:
Catalasa (-), Esculina (-), Ureasa (-), Gelatinasa (-)
Hemólisis (+), Pyrazinamidasa (+), DNAsa (+), CAMP reversa (+)


Arcobacter
La identificación se realiza según recomendaciones del fabricante. No existe validación molecular en nuestra experiencia.
SCORE ≥2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE <1,7 = no identifica
 


Arthrobacter
Debido a limitada experiencia, se recomienda informar sólo género: Arthrobacter sp. con valor de score ≥1,5.
Es necesaria la secuenciación parcial del gen 16S ARNr para una completa identificación.


Bacillus
Aclaración: puede dar fallas en la identificación según el grado de esporulación; utilizar cultivos frescos.
Se emplean los criterios recomendados por el fabricante:
SCORE ≥2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE <1,7 = no identifica.
Para la identificación por MALDI-TOF de especies de Bacillus, es importante seguir las siguientes recomendaciones para informar los resultados correctamente. La especie B. beringensis debe reportarse como Robertmurraya beringensis (Yu et al. 2012) Gupta et al. 2020. B. carboniphilus y B. licheniformis pueden informarse a nivel de especie, pero en el caso de B. licheniformis, solo si se confirma que es anaerobio positivo.
Para las especies pertenecientes al grupo Bacillus cereus (B. cereus, B. cytotoxicus, B. thuringiensis, B. mycoides, B. pseudomycoides y B. weihenstephanensis), deben reportarse como parte de este grupo debido a su cercanía fenotípica y genotípica. Las especies del grupo Bacillus subtilis (B. subtilis, B. atrophaeus, B. mojavensis, B. vallismortis, y B. sonorensis) se informan de igual forma. La especie B. amyloliquefaciens debe informarse como "grupo operacional B. amyloliquefaciens" y se entiende que incluye B. amyloliquefaciens, B. siamensis y B. velezensis.
Los miembros del grupo Bacillus pumilus (B. pumilus, B. safensis, y B. altitudinis) y del grupo Bacillus circulans (B. circulans, B. firmus, B. lentus y B. coagulans) también deben reportarse como grupos para mantener la coherencia en la identificación.
En el caso de B. clausii, debe informarse como Alkalihalobacillus clausii (Patel S et al. 2020) y se recomienda confirmación mediante 16S ARNr o gyrB. Para B. halmapalus, B. horikoshii, B. jeotgali y B. simplex, se deben reportar como Bacillus sp. y confirmar con el gen gyrB.



Bacteroides
Se recomiendan los criterios aceptados en la bibliografía:
SCORE ≥1,8 = identificación a nivel de especie SCORE 1,6-1,79 = identificación a nivel de género SCORE <1,6 = no identifica.
Según Jorgensen et al. (2015) la identificación a nivel de especie es correcta. Sin embargo sugerimos discriminar la identificación en ciertos casos:
Cuando MALDI-TOF arroje un resultado de Bacteroides ovatus, se deberá informar Bacteriodes ovatus/ xylanisolvens debido a su estrecha similitud y a la ausencia de este último en la base de datos del equipo, o realizar pruebas manuales útiles en su diferenciación (Ver Tabla 19).
Por otra parte, MALDI-TOF puede dar como Bacteroides stercoris, las especies Bacteroides fragilis/clarus (
Cuando MALDI-TOF arroje un resultado de Bacteroides vulgatus, se deberá informar Bacteroides vulgatus/dorei debido a su estrecha similitud y a la ausencia de este último en la base de datos del equipo. Del mismo modo, cuando arroje un resultado como Bacteroides thetaiotaomicron se deberá informar como B. thetaiotamicron/faecis, ya que poseen similitud pero el número de espectros de B. faecis es menor que B. thetaiotaomicron.
Diferenciación fenotípica de especies de Bacteroides spp. con dificultades por MALDI- TOF.
 

Especie	
Indol	

Catalasa	
α-Fucosidasa	

Arabinosa	

Trehalosa	
Xilosa
B. stercoris	+	V	V	-	-	+
B. fragilis	-	+	+	-	-	+
B. clarus	+	-	-	-	W	+
B. ovatus	+	+	+	+	+	+
B.
xylanisolvens	-	-	+	+	+	+
Símbolos: V, variable; W, débil.


Bartonella
Debido a escasa experiencia se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE <1,7 = no identifica
Las especies del género Bartonella son consideradas patógenos emergentes; poseen un ciclo natural que incluye infección persistente intraeritrocitaria en un huésped que actúa como reservorio. Los vectores artrópodos transmiten la bacteria entre el reservorio y un huésped susceptible, incluyendo humanos.
De las 37 especies reconocidas oficialmente, las más comunes como patógenos humanos son: Bartonella bacilliformis, Bartonella quintana y Bartonella henselae.
Las especies de Bartonella crecen muy lentamente, requiriendo desde 7 días hasta 6 semanas de incubación.
De acuerdo a la bibliografía, MALDI-TOF identifica correctamente las especies del género luego de incorporar a la base de datos comercial, perfiles proteicos (MSP) de cepas de referencia.
Existe un único espectro de referencia en la base de datos del equipo y corresponde a Bartonella japonica.


Bergeyella
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE <1,7 = no identifica



Bifidobacterium
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género
SCORE < 1,7 = no identifica
Según datos de escasa experiencia de aislados secuenciados, Bifidobacterium scardovii
puede ser informada a nivel de especie con valores de score ≥ 1,7.



Bilophila
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica.
Existen dos espectros de B. wadsworthia que es la única especie del género.



Bordetella
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica
B.	bronchiseptica, B. pertussis y B. parapertussis no son correctamente diferenciadas por Espectrometría de masas aplicando los criterios del fabricante.
Se recomienda utilizar los siguientes criterios:
i.	Score ≥2 + Categoría de Consistencia A = identificación a nivel de especie.
ii.	Score ≥2 + Categoría de Consistencia B = aplicar divergencia del 5% para informar la especie, de lo contrario informar sólo el género.
iii.	Score 1,7-1,9 y top ten con una sola especie = identificación a nivel de especie.
iv.	Score 1,7-1,9 y top ten con distintas especies= aplicar divergencia del 5% para informar la especie, de lo contrario informar sólo el género.
Consistencia A: la primera especie identificada aparece en color verde, el resto de los pocillos en color verde corresponden a esa misma especie. Si aparecen resultados en amarillo corresponden al menos al mismo género que la primera.
Consistencia B: la primera especie identificada aparece en verde o amarillo; otras especies del género pueden aparecer en verde o amarillo. No se cumple el criterio de identificación al nivel de especie.


Pruebas fenotípicas de las especies de Bordetella spp.

Ensayo	B. pertussis	
B. parapertussis	B. bronchiseptica	B. avium	B. hinzii	
B. holmesii	
B. petrii	B. trematum	B. bronchialis	B. flabilis	B. sputigena
Oxidasa	+	-	+	+	+	-	+	-	+	+	+
Catalasa	+	+	+	+	-	+	+	+	+	+	+
Movilidad	-	-	+	-	+	-	-	+	+	+	+
Pigmento	-	marrón	-	-	-	marrón	amarillo	-	-	-	-
Desarrollo en Mac Conkey	nd	+	+	+	+	+	+	+	+	+	+
Urea	-	+	+	-	-	-	-	-	nd	nd	nd
Símbolos: w, débil; nd, no determinado.
Algoritmo de PCR propuesto por el LNR (Servicio Bacteriología Clínica) para la identificación y la confirmación de las especies de Bordetella relacionadas a Coqueluche:
•	B. pertussis:
Blancos de amplificación utilizados: IS481 + ptxS1 ó IS481 + ptxP
•	B. parapertussis:
Blancos de amplificación utilizados: pIS1001 + ptxS1
•	B. holmesii:
Blancos de amplificación utilizados: IS481 + hIS1001
Para la confirmación de estas especies es necesario utilizar al menos dos blancos de amplificación distintos.
 

Brevibacillus
Se recomienda informar sólo a nivel de género. Su significación como patógeno humano es desconocida


Brevibacterium
En general MALDI-TOF funciona correctamente con este tipo de microorganismos; pero puede no identificar según el estado del cultivo.
Debido a limitada experiencia, se recomienda informar sólo a nivel de género: Brevibacterium sp. con valor de score ≥1,5, excepto en Brevibacterium casei que puede identificarse a nivel de especie con score ≥1,7 (Barberis et al., 2014).
Es necesaria la secuenciación parcial del 16S ARNr para una completa identificación.


Brevundimonas
Se recomienda informar con los siguientes criterios:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,9 = identificación a nivel de género SCORE < 1,7 = no identifica
Para la identificación de especies en este grupo, se deben tener en cuenta ciertas características bioquímicas específicas:
B. bullata es negativa para ácido de maltosa, lo cual puede ser un factor útil para diferenciarla.
B. diminuta, que produce un pigmento marrón, es negativa tanto para esculina como para ácido de maltosa.
 B. vancanneytii es negativa para esculina pero positiva para ácido de maltosa, lo que puede orientar su identificación en relación con otras especies.
B. vesicularis, que presenta un pigmento amarillo naranja, es positiva tanto para esculina como para ácido de maltosa, diferenciándose de otras especies que no comparten este perfil.
Para las demás especies (B. abyssalis, B. albigilva, B. aurantiaca, B. intermedia, B. nasdae y B. terrae), no se describen características adicionales específicas en esta tabla, por lo que pueden requerir pruebas adicionales para su correcta identificación.



Brucella
El servicio de Brucelosis del LNR transfirió base de datos propia y protocolos de extracción e INACTIVACIÓN, además de las recomendaciones acerca de como informar. Contactarse con: gescobar@anlis.gob.ar, ccelestino@anlis.gob.ar para solicitarlo.


Burkholderia
MALDI-TOF diferencia correctamente Burkholderia vietnamensis, Burkholderia seminalis y Burkholderia gladiolii.
En caso de no lograrse el 10% de divergencia entre las especies del complejo, se deberá informar Burkholderia complejo cepacia, ya que las mismas son correctamente diferenciadas de microorganismos con fenotipos similares (Ralstonia, Cupriavidus, Pandoraea spp.)
Asimismo para la completa identificación a nivel de especie es necesaria la secuenciación del gen recA.

Campylobacter
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica



Capnocytophaga
En base a datos obtenidos basados en nuestra experiencia, se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica
Para la completa identificación a nivel de especie se utiliza la secuenciación parcial del gen 16S ARNr o rpoB.






Cardiobacterium
En base a datos obtenidos basados en nuestra experiencia, se recomienda informar la identificación sólo a nivel de género con valor de score >1,5. El género incluye dos especies A diferencia de C. hominis, C. valvarum crece más lentamente, es no hemolítico en agar sangre de carnero y no utiliza sacarosa, maltosa ni manitol.



Cedecea
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica

Cellulomonas
Se recomienda informar la identificación sólo a nivel de género con valor de score >1,5.
La identificación definitiva de estas especies suele llevarse a cabo mediante biología molecular.


Cellulosimicrobium
Por ser bacilos gram positivos del grupo pigmentado, MALDI-TOF puede identificarlos a nivel de género con valor de score > 1,5.
El género Cellulosimicrobium incluye bacterias Gram positivas que suelen habitar en el suelo y ambientes acuáticos, y algunas especies pueden asociarse a infecciones oportunistas en humanos, particularmente en individuos inmunocomprometidos. La identificación de especies de Cellulosimicrobium requiere el análisis de su capacidad de crecimiento a distintas temperaturas, movilidad y características bioquímicas específicas. Se sugiere informar a nivel de género y en caso de ser relevante, agregar pruebas bioquímicas o confirmar en laboratorio de referencia. La secuenciación del gen del 16S ARNr y rpoB no logran discriminar a nivel de especie.
Para la identificación de C. cellulans, C. funkei y C. terreum, se pueden seguir las siguientes pautas:
Crecimiento a 35°C y 42°C: C. cellulans y C. funkei crecen a ambas temperaturas, mientras que C. terreum no crece a 35°C ni a 42°C, lo cual es un dato diferenciador importante.
Movilidad: C. funkei muestra movilidad en preparación en fresco, mientras que C. cellulans y C. terreum son no móviles.
Producción de ácido a partir de rafinosa: Solo C. cellulans puede producir ácido a partir de rafinosa (resultado positivo o variable).
Asimilación de compuestos específicos:
C. cellulans es positiva para D-xilosa y negativa para glicerol (glyc) y metil-D-glucosa (MDG).
C. funkei asimila glicerol y MDG, pero no D-xilosa.
C. terreum es positiva para MDG, pero negativa para glicerol y D-xilosa.


Chromobacterium
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica




Chryseobacterium
Se han aceptado valores de score para una identificación confiable:
SCORE >1,9 = identificación a nivel de especie SCORE 1,6-1,89 = identificación a nivel de género SCORE <1,6 = no identifica.
Para las especies Chryseobacterium gleum y Chryseobacterium indologenes se deberá informar como C. gleum / C. indologenes.
Para la identificación y nomenclatura actual de las especies de este grupo, es importante considerar sus nombres actualizados y las limitaciones de identificación por 16S ARNr en ciertos casos.
•	Especies como C. anthropi y C. treverense pueden no diferenciarse claramente de C. haifense y C. solincola respectivamente mediante análisis de 16S ARNr, pero se considera que C. haifense y C. solincola no son patógenas para humanos, lo que ayuda a orientar el diagnóstico.
•	Las especies C. arachidiradicis, C. bovis, C. caeni, C. hispanicum, C. hominis,
C. hungaricum, C. pallidum y C. zeae han sido reclasificadas en el género Epilithonimonas (por ejemplo, Epilithonimonas arachidiradicis para C. arachidiradicis).
•	C. gleum y C. indologenes no pueden diferenciarse entre sí mediante 16S ARNr y pueden necesitar otros métodos confirmatorios para asegurar la correcta identificación.
•	En el caso de C. hominis, es recomendable utilizar el análisis confirmatorio de 16S ARNr debido a la relevancia clínica.
Para las demás especies sin cambios de nomenclatura o comentarios específicos, se mantiene la denominación original, aunque se recomienda verificar actualizaciones taxonómicas en la literatura científica reciente para un reporte preciso.
Las especies del género Chryseobacterium son patógenos emergentes responsables de infecciones en humanos, afectando principalmente a recién nacidos e individuos inmunocomprometidos. Estas bacterias habitan comúnmente en el suelo y en el agua, y con frecuencia colonizan los suministros de agua en hospitales, lo que las convierte en una fuente potencial de infecciones nosocomiales.
Chryseobacterium indologenes es la especie aislada con mayor frecuencia, seguida por Chryseobacterium gleum. Estas bacterias pueden provocar infecciones graves, como meningitis, neumonía, endocarditis, bacteriemia, así como infecciones de piel, tejidos blandos y oculares.
Yadav y col., realizaron un estudio prospectivo de 3 años en un hospital de atención terciaria en el norte de la India para identificar diferentes especies de Chryseobacterium utilizando espectrometría de masas de tiempo de vuelo de desorción/ionización láser asistida por matriz (MALDI-TOF MS) y correlacionarlas clínicamente con patrones de susceptibilidad antimicrobiana.23
El estudio encontró que la rifampicina era el antimicrobiano más eficaz contra C. indologenes, con una susceptibilidad del 75%, seguida de piperacilina-tazobactam (45%).35 La nitrofurantoína también se mostró prometedora en el tratamiento de infecciones del tracto urinario causadas por C. indologenes.56
No existen pautas específicas de pruebas de susceptibilidad antimicrobiana disponibles para el género Chryseobacterium, por lo que los puntos de corte para otros organismos como Pseudomonas aeruginosa y Staphylococcus aureus se utilizaron para la interpretación.

Citrobacter
Se recomienda aplicar los criterios recomendados por el fabricante y para la identificación e informe de especies del género Citrobacter, se recomienda seguir los lineamientos de acuerdo con las características específicas de cada especie y su agrupación en complejos.



Clostridium
El género comprende más de 200 especies anaeróbicas, ocasionalmente aerotolerantes; sin embargo el número de clostridios clínicamente relevantes en infecciones humanas es reducido. Se debe trabajar con cultivos frescos, ya que la esporulación afecta directamente la calidad del espectro. Para mejorar la calidad del espectro obtenido, realizar la técnica de extracción en tubo con etanol/ ácido fórmico recomendada por el fabricante.
Se recomienda aplicar los criterios de interpretación recomendados por el fabricante.
Es importante confirmar especie para C. septicum (asociado a neoplasias gastrointestinales), C. perfringens, C. ramosum, C. innocuum, y C. clostridioforme, generalmente resistentes a antibióticos.
Para la identificación de especies del género Clostridium, se recomienda considerar las siguientes características bioquímicas y morfológicas específicas para cada especie:
•	C. clostridioforme: Esta especie es lactosa positiva y b-NAG negativa.
•	C. innocuum: C. innocuum es sacarolítico y no proteolítico, y es negativo para la prolina aminopeptidasa. No produce esporas de manera habitual; cuando están presentes, se encuentran en posición terminal y son difíciles de observar. Muestra una estructura interna con apariencia de mosaico y es inmóvil. Positivo para esculina y manitol, pero negativo para gelatinasa, lecitinasa, y digestión de leche.




Comamonas
Las especies del género raramente causan enfermedad en el humano. Entre ellas y según la literatura, la más frecuente es C. testosteroni descripta en endocarditis, meningitis y bacteremia asociada a catéter. Sin embargo, de acuerdo a nuestra experiencia, la especie de aislamiento clínico más frecuente es Comamonas kerstersii y nosotros suponemos que en la literatura, las infecciones por C. kerstersii pueden haber estado subestimadas porque en los casos publicados de infecciones por Comamonas, anteriores al año 2013, la identificación de los aislamientos se ha logrado solo mediante métodos automatizados o miniaturizados, que solo cuentan en sus bases de datos con la especie C. testosteroni; además dichos trabajos carecían de la confirmación molecular.
Se recomienda aplicar los criterios recomendados por el fabricante, sin embargo en el caso de Comamonas kerstersii se acepta informar a nivel de especie con valor de score
>1,7.
De no alcanzarse la divergencia del 10% entre las especies, se podrán utilizar la secuenciación del gen 16S ARNr


Corynebacterium
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica
Si se identifica Corynebacterium diphtheriae, Corynebacterium ulcerans y Corynebacterium pseudotuberculosis deben ser derivados al Servicio de Bacteriología Clínica del LNR para la búsqueda de toxinas mediante PCR.
Las especies lipofílicas pueden arrojar scores más bajos. El agregado de 1μl de ácido fórmico puede mejorar la identificación
MALDI-TOF identifica correctamente Corynebacteriun durum (presenta adherencia al agar), C. mucifaciens, C. kroppenstedtii y C. tuberculostearicum.
La metodología recomendada para la completa identificación de la mayoría de las especies corineformes es la secuenciación del gen rpoB.
MALDI-TOF Biotyper identifica la especie C. phoceense como Corynebacterium sp. 901400365 LBK o 901600604 LBK. Por lo tanto frente a un resultado numérico confirmar con las pruebas sugeridas e informar como C. phoceense.
Limitaciones en la identificación para especies de Corynebacterium spp.


ID por MALDI-TOF	Posibles errores en la ID	Confirmación	Informar


C. aurimucosum	
No discrimina
C. aurimucosum /
C. minutissimum	Fenotipia: DNAsa, Hipurato, Tirosina
Secuenciación rpoB	Si colonia cremosa, DNAsa -, Hipu +, Tirosina -:
C. aurimucosum (confirmar con secuenciación)
C. minutissimum	No discrimina
C aurimucosum / C.minutissimum/
C. singulare/
C amycolatum	Fenotipia: DNAsa, Hipurato, Tirosina
Secuenciación rpoB	Si colonia pequeña seca, DNAsa +tardía, Hip-, Tirosina
+: C. minutissimum (confirmar con secueciacion)
C. propinquum	No discrimina
C. pseudodiphteriticum
/ C. propinquum	Urea -/+ Secuenciación rpoB	Si urea –, informar como C. propinquum.
C. amycolatum	Puede confundir con
C. aurimucosum o
C. minutissimum	Fenotipia: aspecto colonia, NO3, tributirina
Secuenciacion rpoB	Si colonica seca, cerosa NO3+, tributirina +: informar como C. amycolatum. Si dichas pruebas son – completar fenotipia y secuenciación.
C. coyleae	Puede confundir con
C. afermentans	LAP
Secuenciación rpoB	LAP+: informar como C. coyleae
C. pseudodiphteriticum	No discrimina
C. pseudodiphteriticum / C. propinquum	Urea + Secuenciación rpoB
Si urea +, no se puede diferenciar con C. propinquum: informar según rpoB
C. simulans	Puede confundir con
C. striatum	CAMP, Etilenglicol
Secuenciación rpoB	Si CAMP-, Etilenglicol –: informar C. simulans
C. striatum	Puede confundir con
C. simulans	CAMP, Etilenglicol


Secuenciación rpoB	Si CAMP+, informar C. striatum. Si CAMP-, etilenglicol +: informar C. striatum
C.phoceense	Puede confundir con
C. aurimucosum o C. glucuronolyticum	Fenotipia	Si Esculina +, NO3+, PYR+  GUR -


Especies de Corynebacterium generalmente multiresistentes:

	C. afermentans ss afermentans
	C. amycolatum
	C. aurimucosum
	C. confusum
	C. coyleae
	C. glucuronolyticum
	C. jeikeium
	C. macginleyi
	C. minutissimum
	C. resistens
	C. striatum
	C. tuberculostearicum
	C. urealyticum
	C. ureicelerivorans

Especies con alta similitud 16S ARNr Marcadores fenotípicos Confirmación

C. afermentans
C. coyleae
C. mucifaciens	C. afermentans sb afermentans: metabolismo fermentativo, CAMP V
C. coyleae: metabolismo oxidativo, CAMP +
C. mucifaciens: colonias mucoides amarillas rpoB
C. aurimucosum
C. minutissimum
C. singulare
C. phoceense	
C. aurimucosum: colonias amarillentas, algunas socaban agar, algunas pigmento gris-negro
C. minutissimum: tirosina +, Urea -
C. singulare: tirosina +, Urea +
C. phoceense: Esc + rpoB
C. propinquum
C.pseudodiphtheriticum	C. pseudodiphteriticum: Urea +
C. propinquum: Urea - rpoB
C. sundsvallense
C. thomssenii Fenotípicamente indistinguibles	rpoB
C. ulcerans
C. pseudotuberculosis	Ambos CAMP reversa +
C. ulcerans: O129 sensible
C. pseudotuberculosis: O129 resistente Pueden tener toxina diphtherica + rpoB
C. xerosis
C. hansenii
C. freneyi	C. xerosis: PAL +, α-glu V, desarrollo 20ºC -, Ferm Glu 42C -
C. hansenii: PAL -
C. freneyi: PAL +, α-glu +, Desarrollo 20ºC +, Ferm Glu 42C +rpoB parcial no discrimina
C. ureicelerivorans
C. mucifaciens	C. ureicelerivorans: urea Rápida, colonia lisa, especie lipofílica
C. mucifaciens: urea -, colonia mucoide amarilla, especie no lipofílica	rpoB parcial no discrimina


Cronobacter
Género en revisión. Se recomienda aplicar los criterios sugeridos por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica
Cronobacter sakazakii (anteriormente conocido como Enterobacter sakazakii) ha sido aislado de sitios clínicos como tracto respiratorio, heridas, líquido cefalorraquídeo (LCR) y heces. Esta especie es de particular preocupación en neonatos, donde se asocia a infecciones graves, como meningitis y sepsis. También puede afectar a individuos inmunocomprometidos, con complicaciones en heridas y tracto respiratorio.
Cronobacter dublinensis se ha aislado de heridas, ojos y sangre. Aunque menos común, esta especie puede causar infecciones localizadas y bacteriemia, particularmente en personas con sistemas inmunitarios comprometidos, como en pacientes hospitalizados o con enfermedades subyacentes.
Cronobacter malonaticus ha sido identificado en muestras clínicas como sangre, heridas, tracto respiratorio y oído. Su aislamiento en sangre sugiere que esta especie puede estar involucrada en infecciones sistémicas, incluidas sepsis, que pueden ser peligrosas en pacientes hospitalizados y en aquellos con condiciones preexistentes que afectan su inmunidad.
Cronobacter muytjensii ha sido aislado de médula ósea y sangre. Su presencia en la médula ósea indica que puede ser una causa de infecciones hematológicas y bacteriemia. Es fundamental su identificación temprana para un tratamiento adecuado, especialmente en pacientes con inmunosupresión o condiciones debilitantes.
Cronobacter turicensis se ha aislado de sangre, lo que sugiere su implicancia en infecciones sistémicas graves como sepsis. Aunque menos frecuente, su capacidad de causar bacteriemia requiere atención médica inmediata, particularmente en individuos vulnerables, como neonatos y personas inmunocomprometidas.


Cupriavidus
Limitada experiencia en el género.
Fenotípicamente similar a Ralstonia sp. Las especies del género que causan enfermedad en el humano han sido aisladas, en su gran mayoría, de cultivos de esputo en pacientes fibroquísticos y bacteremia asociada a catéter.
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica



Cutibacterium (ex Propionibacterium)
Se han aceptado valores de score para una identificación confiable en base a resultados sobre las especies C. avidum y C. acnes:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica
Se sugiere completar con las pruebas fenotípicas de la siguiente tabla para diferenciar Cutibacterium acnes (Indol positivo) de otras especies aisladas de muestras clínicas (Indol negativo). Las especies de Cutibacterium presentan características fenotípicas clave que pueden ayudar a su identificación y diferenciación en el laboratorio.
Cutibacterium acnes (anteriormente conocida como Propionibacterium acnes) muestra una beta hemólisis débil (++) y es indol positivo, lo que indica su capacidad para producir indol a partir de ciertos sustratos. Esta especie también reduce nitratos (NO3 +) y es pirazinamidasa positiva (PYR +), lo que ayuda a diferenciarla de otras especies en el género. No fermenta esculina (Esc -), lo cual es relevante para su identificación.
Cutibacterium avidum presenta una fuerte beta hemólisis (+++) y es indol negativa, lo cual la distingue de otras especies como C. acnes. Esta especie no reduce nitratos (NO3
-) pero es esculina positiva (Esc +) y pirazinamidasa positiva (PYR +), lo que contribuye a su perfil fenotípico único.
Cutibacterium granulosum, por otro lado, no presenta hemólisis en agar (beta hemólisis
-) y es negativa para los tests de indol, nitratos y esculina. Esta especie tiene una identificación fenotípica más limitada, sin características claras de reactividad en estos ensayos, lo que hace necesario el uso de métodos adicionales para su identificación.
Cutibacterium namnetense tiene una beta hemólisis débil (+), es indol positiva y reduce nitratos (NO3 +), pero no fermenta esculina (Esc -). No es positiva para la pirazinamidasa (PYR -), lo que puede ayudar a diferenciarla de especies como C. acnes.
Cutibacterium modestum presenta una reacción negativa para los tests de beta hemólisis y nitratos, con resultados indeterminados (nd) para esculina y pirazinamidasa. Sin embargo, es pirazinamidasa positiva (PYR +), lo que puede servir para su identificación en contextos clínicos.


Cutibacterium acnes posee tres subespecies: C. acnes ss acnes; C. acnes ss defendens, C. acnes ss elongatum.
Diferenciación fenotípica de especies de Cutibacterium spp.

Especie	Beta hemólisis	Indol	NO3	Esc	PYR
C. acnes	++	+	+	-	+
C. avidum	+++	-	-	+	+
C. granulosum	-	-	-	-	nd
C. namnetense	+	+	+	-	-
C. modestum	-	-	nd	nd	+
nd: no determinado


Delftia
Se recomienda aplicar los criterios recomendados por el fabricante y se sugiere informar: Delftia acidovorans sensu lato (incluye las especies D. acidovorans y D. lacustris (Delftia tsuruhatensis), indistinguibles mediante 16S ARNr).



Dermabacter
Dermabacter hominis:
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica.
Sin embargo, debido a una pobre representatividad de MSPs de las otras especies aisladas en humanos, es posible que se requiera el agregado de pruebas bioquímicas si el aislamiento es jerarquizado.
Dermabacter hominis produce fosfatasa alcalina (FAL +) pero no produce α-gal (αGal
-), Tampoco es positiva para tripsina ni para glicerol, lo que contribuye a su perfil bioquímico distintivo.
Dermabacter jinjuensis también es positiva para FAL y muestra actividad α-gal (αGal
+), lo que la diferencia de Dermabacter hominis. Esta especie no tiene datos disponibles (ND) para la prueba de tripsina, pero es negativa para glicerol, lo cual es relevante para su identificación.
Dermabacter vaginalis, es FAL negativa pero muestra una actividad débil de α-gal (αGal W). Esta especie es positiva tanto para tripsina como para glicerol, lo que la diferencia de las otras dos especies en este conjunto.
Tabla 52. Caracteres fenotípicos diferenciales para especies de Dermabacter spp.

Especie	FAL	αGal	Tripsina	Glicerol
D. hominis	+	-	-	-
D. jinjuensis	+	+	ND	-
D. vaginalis	-	W	+	+
Símbolos: W, positivo débil; ND, no determinado.



Desulfovibrio
Son microorganismos reductores de sulfatos, residentes del tracto gastrointestinal pero pueden ser hallados de manera infrecuente en especímenes clínicos; bacteremia e infecciones abdominales en pacientes inmunocomprometidos.
La baja tasa de recuperación de los integrantes de este género en muestras clínicas puede deberse a su lento desarrollo y a la necesidad de contar con herramientas moleculares para su identificación.
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica
Las limitaciones en la identificación son debidas a la ausencia o a la escasa representación (ejemplo: D. desulfuricans) del perfil de proteínas del microorganismo en cuestión en la base de datos comercial.


Tabla 53. Traducción de especies de Desulfovibrio spp.

Especie	Presencia en BD	Nro de MSPs en BD	Asialmiento humano
D. desulfuricans	SI	3	SI
D. fairfieldensis	NO		SI
D. piger	SI	4	SI
D. simplex	SI	2	NO
D. vulgaris	NO		SI
Diferenciación fenotípica de especies clínicas de Desulfovibrio spp.

Especie	Reducción de NO3	Catalasa	Indol	Ureasa
D. desulfuricans	+	-	-	+
D. fairfieldensis	+	+	-	-
D. piger	-	-	-	-
D. vulgaris	-	-	+	-



Dialister
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 1,9 = identificación a nivel de especie SCORE 1,7-1,9 = identificación a nivel de género SCORE < 1,7 = no identifica

Dietzia
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica


Dolosicoccus
No hay perfil de referencia en base de datos comercial.


Dolosigranulum
Existen dos perfiles de referencia en la base de datos comercial, correspondientes a
Dolosigranulum pigrum.
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica
Se sugiere confirmar la identificación de este patógeno raro o infrecuente mediante Biología Molecular, según normas del CLSI.

Dysgonomonas
De acuerdo a nuestra experiencia se recomienda informar a nivel de género con valor de score >1,7. Tres especies de aislamiento humano: D. hofstadii, D. mossii y D. massiliensis no están bien representadas en las bases de datos comerciales, y puede identificarse erróneamente Dysgonomonas gadei
No es posible la diferenciación fenotípica entre las especies.
Para la completa identificación se recomienda secuenciar el gen 16S ARNr.


Edwarsiella
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica



Eggerthella
Las especies del género Eggerthella y Paraeggerthella han sido recuperadas de una gran variedad de infecciones humanas. E. lenta (ex Eubacterium lentum) es un patógeno intrabodominal frecuente. E. lenta, E. sinensis y P. hongkongensis han sido recuperados de hemocultivos y asociados a infecciones con alta tasa de mortalidad.
Son cocobacilos o bacilos cortos gram positivos no esporulados, anaerobios, que se disponen en pares o cadenas cortas.
Características fenotípicas de especies de Eggerthella spp.
Especie	Fermentacion de Glucosa	Catalasa	
Indol	Reduccion de Nitratos	Hidrolisis de esculina	Hidrolisis de Arginina
Eggerthella lenta	-	+	-	+	-	+
Eggerthella sinensis	-	+	-	-	ND	+



Eikenella corrodens
En base a datos obtenidos basados en nuestra experiencia, se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica



Elizabethkingia
Las especies del género suelen ser multirresistentes a los antibióticos de uso común y son agentes causantes de brotes en el área hospitalaria, sobre todo meningitis en recién nacidos y pacientes inmunocomprometidos.
Actualmente el género comprende seis especies (E. anophelis, E. meningoseptica, E.
miricola, E. bruuniana, E. ursingii y E. occulta), de las cuales, hasta el momento, tres tienen importancia médica: E. meningoseptica, E. anophelis y E. miricola.
Luego de la última actualización del software, MALDITOF identifica correctamente las especies Elizabethkingia meningoseptica, Elizabethkingia miricola y Elizabethkingia anophelis.
La identificación se realiza según recomendación del fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE <1,7 = no identifica
Es importante que los microbiólogos clínicos estén actualizados con respecto a este patógeno emergente. Para laboratorios de microbiología de baja complejidad, el aislamiento a partir de un líquido estéril de un bacilo gramnegativo no fermentador multirresistente que es oxidasa positivo, resistente al colistín y que no presenta pigmento o con pigmento amarillo muy pálido, puede ser confirmado fácilmente como probable Elizabethkingia sp. si produce una prueba positiva de indol y es inmóvil.


Características fenotípicas de especies de Elizabethkingia spp. aisladas de muestras clínicas.



Especie	
Citrato	Nitratos	MConkey	
Ureasa	
Ácido de celobiosa	
Ácido de melibiosa	Ácido de melezitosa
E. meningoseptica	V		-/v	v	-/v	+	ND
E. miricola	+	+	-	+	v	-	ND
E. anophelis	-	v	v	v	v	v	ND
E. bruuniana	V	-	ND	ND	v	ND	-
E. occulta	-	+	+	+	+	ND	-
E. ursingii	v	-	v	v	-	ND	-
Símbolos: V, variable.


Empedobacter brevis
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica



Enterobacter
La literatura demuestra que ambas plataformas MALDI-TOF tienen una muy buena sensibilidad y especificidad para identificar las especies aisladas con mayor frecuencia como Enterobacter cloacae y Enterobacter (Klebsiella) aerogenes, sin embargo hay trabajos que documentan que la resolución de MALDI-TOF MS es inadecuada para delinear las especies que integran el complejo Enterobacter cloacae (Enterobacter cloacae, Enterobacter asburiae, Enterobacter hormaechei, Enterobacter kobei, Enterobacter ludwigii y Lelliottia (Enterobacter) nimipressuralis). A continuación, se presenta una guía sobre cómo informar las especies del género Enterobacter y sus cambios taxonómicos o comentarios relevantes para la identificación en el laboratorio clínico:
Enterobacter aerogenes: Actualmente se debe informar como Klebsiella aerogenes. Enterobacter asburiae: Debe informarse como parte del Complejo Enterobacter cloacae. Enterobacter bugandensis: No se han especificado cambios taxonómicos relevantes.
Enterobacter cancerogenus: Es sinónimo de Enterobacter taylorae, por lo que se debe informar como tal.
Enterobacter cloacae: Se debe informar como parte del Complejo Enterobacter cloacae.
Enterobacter cloacae ssp. cloacae: Al igual que Enterobacter cloacae, se debe informar como parte del Complejo Enterobacter cloacae.
Enterobacter cloacae ssp. dissolvens: También se debe informar como parte del Complejo Enterobacter cloacae.
Enterobacter hormaechei ssp. hormaechei: Informar como parte del Complejo
Enterobacter cloacae.
Enterobacter kobei: Se debe informar como parte del Complejo Enterobacter cloacae.
Enterobacter ludwigii: Al igual que otras especies del Complejo Enterobacter cloacae, debe ser informado en este grupo.
Enterobacter xiangfangensis: Es parte del subgrupo Enterobacter hormaechei subsp. xiangfangensis y debe informarse como parte del Complejo Enterobacter cloacae.
Kosakonia cowanii: Anteriormente conocido como Enterobacter cowanii, debe informarse como Kosakonia cowanii.
Kosakonia radicincitans: Anteriormente Enterobacter radicincitans, debe ser informado como Kosakonia radicincitans y no tiene aislamiento humano.
Lelliottia amnigena: Es conocido anteriormente como Enterobacter amnigenus y debe informarse como Lelliottia amnigena.
Lelliottia nimipressuralis: Anteriormente Enterobacter nimipressuralis, se debe informar como Lelliottia nimipressuralis, y se incluye en el Complejo Enterobacter cloacae.
Metakosakonia massiliensis: Anteriormente Enterobacter massiliensis, se debe informar como Metakosakonia massiliensis, con aislamiento humano.
Pantoea agglomerans: Anteriormente Enterobacter agglomerans, se debe informar como Pantoea sp..
Pluralibacter gergoviae: Anteriormente Enterobacter gergoviae, debe informarse como Pluralibacter gergoviae.


Enterococcus
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE < 1,5 = no identifica



Erysipelothrix
Se recomienda aplicar los criterios recomendados por el fabricante: Identifica correctamente a nivel de especie, con score>2,0.
 

Escherichia
MALDI-TOF identifica como E.coli las especies de Shigella spp. La necesidad de complementar la identificación con pruebas bioquímicas dependerá del tipo de muestra y de la epidemiología local


Eubacterium
El género permanece pobremente definido, pero las especies que lo integran son comúnmente aisladas en infecciones de la cavidad oral. Desarolla en particular cuando se emplean medios enriquecidos y tiempos de incubación prolongados.
Se recomienda aplicar los criterios recomendados por el fabricante:
Las limitaciones se deben en su mayoría a la ausencia del perfil de proteínas del microorganismo en cuestión en la base de datos comercial.

Exiguobacterium
Por ser bacilos Gram positivos de grupo pigmentado, MALDI-TOF puede identificar a nivel de género con valor de score > 1,5.
La especie más frecuente es Exiguobacterium acetylicum.
Exiguobacterium aurantiacum ha sido aislada en sólo seis ocasiones a lo largo de 10 años por los centros de referencia.
Pruebas fenotípicas útiles en la diferenciación de las especies de Exiguobacterium spp.

Especie	Oxidasa	DNAsa	Xilosa	Comentario
Exiguobacterium acetylicum	+	-	-	Pigmento amarillo-oro
Exiguobacterium aurantiacum	-	+	+	Sensible a todas las drogas


Facklamia
El género está estrechamente relacionado con Globicatella, pero es fenotípica y filogenéticamente distinto. Las cepas de las cuatro especies de Facklamia aisladas de humanos han sido recuperadas a partir de sangre, heridas, tracto genitourinario y un caso de corioamnionitis.
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica


Finegoldia magna
Entre los cocos gram positivos anaerobios, se considera la especie más patogénica y ha sido aislada de una gran variedad de sitios de infección (piel, tejido óseo, úlceras, abscesos, infecciones prostéticas). Los múltiples hallazgos sugieren que la significación clínica de Finegoldia magna está subestimada.
Se recomienda aplicar los criterios recomendados por el fabricante:


Francisella
Francisella tularensis es el agente causal de la tularemia, una enfermedad aguda y fatal en animales y humanos. La infección humana ocurre por la mordedura de un artrópodo, contacto con un animal infectado, o por ingestión de agua o alimentos contaminados. Argentina es un pa[is libre de tularemia hasta el momento.
El género está compuesto además por otras especies poco conocidas y consideradas ambientales y/o patógenos oportunistas. Mientras que Francisella noatunensis y Francisella halioticida infectan y causan muertes en peces; Francisella novicida y Francisella philomiragia están asociadas al agua salada y solo aparecen en infecciones oportunistas infrecuentes en individuos inmunocomprometidos.
Las infecciones humanas causadas por Francisella philomiragia son muy poco frecuentes; afectando pacientes con enfermedades de base, principalmente enfermedad granulomatosa crónica (EGC). Es resistente a cotrimoxazol, hecho relevante ya que es el antibiótico de uso profiláctico en pacientes con EGC
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica




Fusobacterium
Se recomienda informar a nivel de género a excepción: F. nucleatum y F. naviforme deben informarse como F. nucleatum/naviforme. En base a nuestra experiencia, la identificación es correcta para las especies F. necrophorum y F. mortiferum.


Gardnerella
G. vaginalis es la única especie representada en las Bases de Datos comerciales. Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE < 1,5 = no identifica
Todos los aislados de G. vaginalis son uniformemente sensibles a SPS (polianetolsulfonato de sodio): halo >10 mm cuando se usan discos y halo > 12 mm cuando se usan tabletas. Se sugiere realizar esta prueba confirmatoria.
Si el aislamiento es recuperado de un proceso invasivo, se recomienda derivar a laboratorio de referencia para confirmar la especie


Gemella
Se han aceptado valores de score para una identificación confiable:
SCORE >1,70 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,50 = no identifica
Pruebas bioquímicas para la diferenciación de especies de Gemella spp.

Especie	PYR	Hipurato	FAL	Acido de Maltosa	Acido de Manitol	Acido de Sorbitol
G. asaccharolytica	-	+	-	-	-	-
G. bergeri	+	-	-	-	+	-
G. haemolysans	+	-	+	+	-	-
G. morbillorum	+	-	-	+	+	+
G. parahaemolysans	+	-	+	+	-	-
G. sanguinis	+	-	+	+	+	+
G. taiwanensis	+	-	+	+	+	+

Globicatella
Globicatella sanguinis ha sido aislado de especímenes clínicos, implicado en casos de bacteremia, infecciones urinarias y meningitis. La segunda especie del género, Globicatella sulfidifaciens, se ha recuperado en infecciones purulentas en animales domésticos. Se recomienda informar a nivel de género o agregar pruebas bioquímicas adicionales para confirmar la especie . Globicatella sanguinis puede distinguirse de Globicatella sulfidifaciens mediante pruebas bioquímicas. Globicatella sulfidifaciens: Esta especie presenta un resultado negativo para PYR, βGal, Manitol, y Ribosa, mientras que tiene un resultado positivo para βGur. Globicatella sanguinis: Es positiva para PYR, βGal, Manitol, y Ribosa, pero negativa para βGur.


Características fenotípicas de especies del género Globicatella spp.

Especie	PYR	βGal	βGur	Manitol	Ribosa
G. sulfidifaciens	-	-	+	-	-
G. sanguinis	+	+	-	+	+
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 =	identificación a nivel de especie (confirmar con pruebas bioquímicas)
SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica


Gordonia
MALDI-TOF identifica correctamente a nivel de género. Se recomienda informar Gordonia sp. con valor de score >1,5. Para mejorar la identificación se recomienda método de extracción en tubo



Granulicatella
Los organismos del género Abiotrophia y Granulicatella son conocidos como variantes nutricionales de Streptococcus (VNS).
La prueba del satelitismo es fundamental para la identificación de ambos géneros aunque debemos tener presente que esta característica no es solo propia de Abiotrophia y Granulicatella dado que también puede estar presente en cepas de otros géneros como Ignavigranum, Helcococcus, Gemella, entre otros.
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica

Caracteristicas fenotípicas de especies del género Granulicatella spp.

Especie	βGur	ADH	Hipurato	Acido de Sacarosa	Acido de Trehalosa
G. adiacens	+	-	-	+	-
G. balaenopterae	-	+	-	-	+
G. elegans	-	+	V	+	-
Símbolos: V, variable.


Haemophilus
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica


Hafnia
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica
H. paralvei es la denominación actual de la otrora denominada Hafnia alvei grupo de hibridización 2.
Ambas especies se pueden aislar con frecuencia a partir de muestras clínicas, ya que producen una toxina citolítica Vero, aunque las cepas de H. alvei tienen más probabilidades de ser toxigénicas que H. paralvei.
 





Los aislamientos de Hafnia pueden ser inequívocamente asignados a la especie correcta (H. alvei o H. paralvei) en base a las pruebas bioquímicas que se detallan a continuación. La identificación de especies del género Hafnia se puede realizar a través de su capacidad para utilizar malonato y citrato.
Hafnia alvei: Esta especie es positiva para la utilización de malonato y negativa para citrato. Estos resultados fenotípicos son clave para diferenciar H. alvei de otras especies en el género.
Hafnia paralvei: A diferencia de H. alvei, H. paralvei es negativa para la utilización de malonato y presenta una reacción positiva para citrato. Este patrón de reactividad también es característico de la especie y se utiliza para su identificación precisa en el laboratorio.
Identificación de especies de Hafnia spp.

Especie	Utilización de Malonato	Citrato
H. alvei	+	-
H. paralvei	-	V
Símbolos: V, variable.

Halomonas
Informar según criterios del fabricante sólo a nivel de género. El género Halomonas es incluye especies de bacterias halofílicas y alcalifílicas, que son típicamente aisladas de ambientes con alta concentración de sal. A pesar de que las especies de Halomonas se encuentran principalmente en ambientes marinos o salinos, algunas han demostrado capacidad de crecimiento en medios carentes de alta concentración salina, como el suero sanguíneo y el dializado. Esto indica que pueden sobrevivir en ambientes menos salinos, lo que amplía su potencial patogénico.
En términos clínicos, la presencia de Halomonas en textos estándar de microbiología médica ha sido históricamente subestimada, lo que ha limitado el reconocimiento de su potencial patógeno. Aunque se había reportado un caso aislado de infección por mordedura de pez atribuida a Halomonas venusta, en los últimos años han emergido informes sobre infecciones bacteriémicas en neonatos causadas por Halomonas phocaeensis en Túnez. Además, un estudio que buscaba firmas bacterianas en muertes inexplicadas identificó una especie no especificada de Halomonas en sangre, sugiriendo la posibilidad de que Halomonas esté implicada en enfermedades críticas de origen no diagnosticado. También ha sido responsable en brotes de bacteriemia en unidades de diálisis.
La capacidad de Halomonas para formar biopelículas y su resistencia a las condiciones de alta salinidad y alcalinidad son factores clave en su supervivencia y posible patogenicidad, particularmente en ambientes hospitalarios y en pacientes inmunocomprometidos o en diálisis. Estos hallazgos resaltan la necesidad de una mayor vigilancia y una mejor comprensión de las especies de Halomonas en el contexto clínico, particularmente en infecciones nosocomiales y en pacientes sometidos a tratamientos médicos prolongados como la hemodiálisis.




Hathewaya (ex Clostridium)


Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica





Helcococcus
Helcococcus kunzii puede ser aislado de piel y de infecciones de hueso sobre todo de las extremidades inferiores, como el pie. Su significancia clínica es difícil de interpretar ya que suele ser un agente colonizante. La habilidad de Helcococcus kunzii de convertirse en un patógeno oportunista es sugerida cuando se recupera como único organismo o predominante en infecciones mamarias, quistes sebáceos, infecciones protésicas, bacteremia y empiema. Helcococcus sueciensis y Helcococcus pyogenes han sido aislados únicamente de una infección de hueso y protésica, respectivamente.
Helcococcus seattlensis ha sido recuperado de un paciente con urosepsis.


Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica
Pueden aumentar los valores de score cuando se realiza el método del ácido fórmico y la técnica de extracción proteica.



Helicobacter
Se recomienda informar a nivel de género con valor de score ≥1,7.
Es necesaria la ampliación de la Base de datos comercial para la completa identificación a nivel de especie; también se puede llevar a cabo la secuenciación del gen específico hsp60. Se recomienda consultar en el manual sobre los nichos y la posible implicancia clínica de las diferentes especies de Helicobacter.



Especie	Comentarios
 




H. acinonychis	Especie aislada de grandes felinos
H. anseris	Aislado de gansos
H. aurati	Aislado de hamsters
H. baculiformis	Aislado de mucosa de gato
H. bilis	Posible asociación con cáncer de colon
H. bizzozeronii	Aislado de mucosa gástrica humana
H. bovis	Aislado de ganado
Nomenclatura no valida
H. brantae	Aislado de gansos
H. canadensis	Aislado de pacientes con diarrea
H. canis	Aislado de muestras humanas
H. cetorum	Aislado de delfines y ballenas
H. cholecystus	Aislado de hamsters
H. cinaedi	Aislamiento humano
H. cynogastricus	Aislado de mucosa gástrica canina
H. equorum	Aislado de heces humanas y de equinos
H. felis	Enfermedad gástrica en el humano y en gatos
H. fennelliae	Aislamiento humano
H. ganmani	Aislado de tejido hepático pediátrico

H. heilmannii	Aislamiento humano
H. hepaticus	Aislamiento humano
H. marmotae	Aislado de gatos
H. mastomyrinus	Aislado de roedores
H.mesocricetorum	Aislado de hamsters
H. muridarum	Aislado de roedores
H. mustelae	Aislado de hurones
H. pametensis	Aislado de cerdos y de aves
H. pullorum	Patógeno zoonótico emergente
H. pylori	
H. rodentium	Aislado de roedores
H. salomonis	Aislamiento animal
H. suis	Aislamiento humano
H. trogontum	Aislamiento humano
H. typhlonius	Aislado de roedores
H. valdiviensis	Potencial patógeno humano
H. winghamensis	Aislamiento humano
Nomenclatura no válida


Herbaspirillum
Sólo informar género.
Herbaspirillum spp. es un patógeno raro que puede ser identificado mediante MALDI- TOF MS a nivel de género, pero la asignación a especie no siempre es confiable. Se recomienda el uso de métodos moleculares para confirmación de la especie. Las especies de Herbaspirillum son altamente susceptibles a los β-lactámicos, con excepción de la ampicilina, y presentan resistencia intrínseca a la colistina. El tratamiento sugerido incluye piperacilina/tazobactam o ceftazidima. Se debe tener en cuenta que los métodos fenotípicos tradicionales pueden llevar a una identificación errónea, por lo que el uso de MALDI-TOF MS es esencial para una identificación precisa del género.



Histophilus somni

Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica
 

Ignavigranum ruoffiae

Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica




Inquilinus limosus
Por presentar un fenotipo mucoso puede presentar problemas para su identificación or el método directo y requerir el agregado de una gota de ácido fórmico sobre la muestra. De no lograr el resultado esperado, se puede intentar el paso de extracción en tubo recomendado por el fabricante.
Se recomienda aplicar los criterios recomendados por el fabricante: SCORE 1,7- |2.0 = identificación a nivel de género y especie SCORE < 1,70 = no identifica
 

Jeotgalicoccus
Se recomienda informar Jeotgalicoccus halotolerans con valor de score >1,7.
Ninguna de las especies descriptas hasta el momento han sido recuperadas de infecciones humanas.


Kerstersia
El género incluye dos especies. Existen más perfiles de referencia para Kerstersia gyiorum, y ninguno para Kerstersia similis. Ambas especies son fenotípicamente indistinguibles, por lo que se sugiere realizar la secuenciación del gen gyrB para su diferenciación.
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,70-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica

Kingella
En base a datos obtenidos basados en nuestra experiencia, se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69= identificación a nivel de género SCORE <1,5= no identifica



Klebsiella
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica

Se recomienda informar Complejo K. pneumoniae. MALDI-TOF no logra discriminar
Klebsiella oxytoca de Raoultella ornithinolytica, debido a la gran similitud que presentan ambos espectros. Se recomienda completar con el perfil bioquímico y/o confirmar la identificación mediante secuenciación del gen rpoB.
Perfil bioquímico diferencial para especies de Klebsiella y Raoultella



Especie	Indol	
ODC	
VP	Malonato	ONPG
Klebsiella oxytoca	+	-	+	+	+
Klebsiella ozaenae	-	-	-	-	V
Klebsiella pneumoniae	-	-	+	+	+
Klebsiella rhinoscleromatis	-	-	-	+	-
Raoultella ornithinolytica	+	+	V	+	+
Raoultella planticola	V	-	+	+	+
Raoultella terrigena	-	-	+	+	+
Símbolos: V, variable.



Kluyvera
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica




Kocuria
El género Kocuria ha ganado relevancia clínica en los últimos años debido a su implicancia como patógeno oportunista, especialmente en pacientes inmunocomprometidos.Se recomienda aplicar los criterios recomendados por el fabricante. Sin embargo, puede no discriminar K. rosea de K. polaris, aunque ésta última no ha sido aislada de humanos. De todas formas, se puede completar la identificación a nivel de especie mediante la secuenciación del gen rpoB. Se recomienda realizar pruebas fenotípicas adicionales.
Kocuria carniphila: Produce un pigmento amarillo. Presenta reducción de nitratos (+), prueba de β-Galactosidasa (+), ureasa negativa (-) y no hidroliza Tween 80 (-).
 





Kocuria kristinae: El pigmento es de color crema a naranja pálido. Tiene una reducción de nitratos variable (V), β-Galactosidasa negativa (-), ureasa negativa (-) y no hidroliza Tween 80 (-).
Kocuria rhizophila: Presenta un pigmento amarillo. No reduce nitratos (-), es negativa para β-Galactosidasa (-), ureasa negativa (-), pero hidroliza Tween 80 (+).
Kocuria rosea: El pigmento es pastel o naranja-rojo. Reduce nitratos (+), es negativa para β-Galactosidasa (-), ureasa negativa (-) y no hidroliza Tween 80 (-).
Kocuria varians: Produce diferentes tonalidades de amarillo oscuro. Reduce nitratos (+), presenta β-Galactosidasa (+), ureasa negativa (-) y no hidroliza Tween 80 (-).



Especie	



Pigmento	Reducción de nitratos	
B GAL	
Ureasa	Hidrólisis de Tween 80
K. carniphila	amarillo	+	+	-	-
K. kristinae	crema a naranja pálido	V	-	V	-
K. rhizophila	amarillo	-	-	-	+
K. rosea	pastel o naranja-rojo	+	-	-	-
K. varians	diferentes tonalidades de amarillo oscuro	+	+	+	-



Kosakonia
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie
SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica



Kurthia
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica





Kytococcus
Se recomienda aplicar los criterios recomendados por el fabricante:


SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica


Lactobacillus
El género ha sufrido cambios taxonómicos con muchas especies reclasificadas en nuevos géneros. Para su identificación se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica
Esta sugerencia está basada en un número reducido de aislados.
Para una inequívoca identificación de especie se requiere métodos moleculares


Cambios taxonómicos del género Lactobacillus

Especie	Cambios taxonómicos
(Zheng et al. 2020)
L. acetotolerans	
L. acidifarinae	Levilactobacillus acidifarinae
L. acidipiscis	Ligilactobacillus acidipiscis
L. acidophilus	
L. agilis	Ligilactobacillus agilis
L. algidus	Dellaglioa algida
L. alimentarius	Companilactobacillus alimentarius
L. amyloliticus	
L. amylophilus	Amylolactobacillus amylophilus
L. amylotrophicus	Amylolactobacillus amylotrophicus	
L. amylovorus	
L. antri	Limosilactobacillus antri
L. apinorum	Apilactobacillus apinorum
L. apodemi	Ligilactobacillus apodemi
L. aviarius	Ligilactobacillus aviarius
L. backii	Loigolactobacillus backii
L. bifermentans	Loigolactobacillus bifermentans
L. brevis	Levilactobacillus brevis
L. buchneri	Lentilactobacillus buchneri
L. casei	Lacticaseibacillus casei
L. cerevisiae	Levilactobacillus cerevisiae
L. coleohominis	Limosilactobacillus coleohominis
L. collinoides	Secundilactobacillus collinoides
L. composti	Agrilactobacillus composti
L. concavus	Lapidilactobacillus concavus				
L. confusus 	Weisella confusa (Collins et al. 1994)				
 	Comentado [FV1]: Esta como Weissella confusa				
L. coryniformis	Loigolactobacillus coryniformis			
L. crispatus				
L. curvatus	Latilactobacillus curvatus			
L. delbrueckii				
L. dextrinicus	Lapidilactobacillus dextrinicus			
L. diolivorans	Lentilactobacillus diolivorans			
L. equi	Ligilactobacillus equi			
L. fabirementans	Lactiplantibacillus fabifermentans			
L. farciminis	Companilactobacillus farciminis			
L. fermentum	Limosilactobacillus fermentum			
L. fornicalis				
L. fructivorans	Fructilactobacillus fructivorans			
L. frumenti	Limosilactobacillus frumenti			
L. fuchuensis	Latilactobacillus fuchuensis			
L. gallinarum	75	
L. gasseri	
L. gastricus	Limosilactobacillus gastricus
L. ghanensis	Liquorilactobacillus ghanensis
L. graminis	Latilactobacillus graminis
L. hammesii	Levilactobacillus hammesii
L. hamsteri	
L. harbinensis	Schleiferilactobacillus harbinensis
L. helveticus	
L. hilgardii	Lentilactobacillus hilgardii
L. hominis	
L. homochiochii	Fructilactobacillus fructivorans
L. hordei	Liquorilactobacillus hordei
L. iners	
L. ingluviei	Limosilactobacillus ingluviei
L. intestinalis	
L. jensenii	
L. johnsonii	
L. kalixensis	
L. kefiri	Lentilactobacillus kefiri
L. kisonensis	Lentilactobacillus kisonensis
L. kitasatonis	
L. kunkeei	Apilactobacillus kunkeei
L. lindneri	Fructilactobacillus lindneri
L. lactis	L. delbrueckii subsp. lactis
(Weiss et al, 1984)
L. malefermentans	Secundilactobacillus malefermentans
L. mali	Liquorilactobacillus mali
L. manihotivorans	Lacticaseibacillus manihotivorans
L . mindensis	Companilactobacillus mindensis
L . mucosae	Limosilactobacillus mucosae
L. murinus	Ligilactobacillus murinus
L .nageli	Liquorilactobacillus nagelii
L . nantensis	Companilactobacillus nantensis
L. nodensis	Companilactobacillus nodensis
L. oeni	Liquorilactobacillus oeni
L. oligofermentans	Paucilactobacillus oligofermentans
L. oris	Limosilactobacillus oris
L. otakiensis	Lentilactobacillus otakiensis
L. panis	Limosilactobacillus panis
L. pantheris	Lacticaseibacillus pantheris
L. parabuchneri	Lentilactobacillus parabuchneri
L. paracasei	
L. paracollinoides	Secundilactobacillus paracollinoides
L. parakefiri	Lentilactobacillus parakefiri
L. paralimentarius	Companilactobacillus paralimentarius
L. paraplantarum	Lactiplantibacillus paraplantarum
L. paucivorans	Levilactobacillus paucivorans
L. pentosus	Lactiplantibacillus pentosus
L. perolens	Schleiferilactobacillus perolens
L. plantarum	Lactiplantibacillus plantarum
L. pobuzihii	Ligilactobacillus pobuzihii
L. pontis	Limosilactobacillus pontis
L. psitacci	
L. rapi	Lentilactobacillus rapi
L. rennini	Loigolactobacillus rennini
L. reuteri	Limosilactobacillus reuteri
L. rhamnosus	Lacticaseibacillus rhamnosus
L. rodentium	
L. rossiae	Furfurilactobacillus rossiae
L. ruminis	Ligilactobacillus ruminis
L. saerimneri	Ligilactobacillus saerimneri
L. sakei	Latilactobacillus sakei
L. salivarius	Ligilactobacillus salivarius
L. sanfranciscensi	Fructilactobacillus sanfranciscensis
L. satsumensis	Liquorilactobacillus satsumensis
L. sharpeae	Lacticaseibacillus sharpeae
	
L. siliginis	Furfurilactobacillus siliginis			
L. similis	Secundilactobacillus similis			
L. spicheri	Levilactobacillus spicheri			
L. suebicus	Paucilactobacillus suebicus			
L. sunkii	Lentilactobacillus sunkii			
L. tucceti	Companilactobacillus tucceti			
L. uli
	Olsenella uli			
				
 	Comentado [M2]: ESTA COMO OLSENELLA ULI

	(Dewhirst et al. 2001)			
L. ultunensis				
L. vaccinostercus	Paucilactobacillus vaccinostercus			
L. vaginalis	Limosilactobacillus vaginalis			
L. versmoldensis	Companilactobacillus versmoldensis			
L.vini	Liquorilactobacillus vini			
L. vitulinus
	Kandleria vitulina			
				
 	Comentado [M3]: EN LA BD esta como KANDLERIA

	(Salvetti et al. 2011)			
L. zeae	Lacticaseibacillus zeae(Liu and Gu 2020) 80	
L. zymae	Levilactobacillus zymae



Lactococcus
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica

Lactococcus spp.
Especie	Cambios taxonómicos	Aislamiento humano
L. garviae		SI
L. lactis		
L. lactis subsp. cremoris	Lactococcus cremoris (Li et al. 2021)	SI
Lactococcus lactis ssp hordniae		NO
L. lactis
subsp. lactis		
Lactococcus lactis ssp tructae	Lactococcus cremoris ssp. trucae (Li et al. 2021)	NO
Lactococcus laudensis		NO
Lactococcus piscium		NO
Lactococcus plantarum		NO
Lactococcus raffinolactis		NO



Leclercia
El género Leclersia incluye tres especies. Dos han sido aisladas en humanos: L. adecarboxylata y L. pneumoniae. Esta última descripta en 2022. Las limitaciones para la identificación de especie dependen de la representaividad de MSPs de ambas especies. Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica



Legionella
En base a limitada experiencia de los LNR en el género, únicamente estamos en condiciones de afirmar que MALDI-TOF identifica correctamente las especies Legionella pneumophila y Legionella micdadei, con valores de score >2.
Leifsonia
Informar solo género


Lelliotia amnigena
Ver Género Enterobacter


Leptotrichia
De acuerdo a nuestra experiencia, recomendamos usar los siguientes criterios para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica.
Debido a que ciertas especies no están representadas, si el aislado es clínicamente relevante, sugerimos confirmar la identificación por métodos moleculares

Especies del género
Leptotrichia	Aislamiento humano	Actualización taxonómica
L. buccalis	SI	
L. goodfellowii	SI	Pseudoleptotrichia goodfellowii
(Eisenberg et al. 2020)
L. hofstadii	SI	
L.
hongkongensis	SI	
L. shahii	SI	
L. trevisanii	SI	
L. wadei	SI	


Leuconostoc
Género vancomicina resistente al igual que Pediococcus, pero Leuconostoc sp. produce gas, es siempre ADH negativa y se dispone en cadenas (a diferencia de Pediococcus spp.) Puede ser aislado de sangre, LCR, líquido peritoneal y heridas, como agente causal de osteomielitis, absceso cerebral, endoftalmitis y bacteriemia.
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica


Listeria:
MALDI-TOF identifica correctamente a nivel de género, pero no discrimina a nivel de especie (generalmente entre L. monocytogenes y L. innocua). El fabricante recomienda la extracción etanólica para la identificación correcta a nivel de especie; sin embargo, es conveniente la confirmación de la especie con pruebas fenotípicas como por ejemplo la prueba de CAMP con S. aureus, para la cual L. monocytogenes muestra sinergia hemolítica en forma de cabeza de alfiler. La listeriosis es una enfermedad de notificación obligatoria en Argentina. Los aislados de casos clínicos en el país deben ser derivados al LNR para vigilancia genómica.Pruebas fenotípicas para diferenciar especies de Listeria spp.
Ensayo	L. grayi	L. innocua	L. ivanovii ss ivanovii	L. ivanovii ss lodoniensis	L. marthii	L. monocytogenes	L. seeligeri	L. welshimeri
 Hemólisis	-	-	++	++	-	+	+	-
CAMP S. aureus
CAMP R. equi	-
-	-
-	-
+	-
+	ND
ND	+
V	+
-	-
-
Hipurato	-	+	+	+	ND	+	ND	ND
Reducción de NO3	V	-	-	-	-	-	ND	ND
Acido de manitol	+	-	-	-	-	-	-	-
Acido de rhamnosa	V	V	-	-	-	+	-	V
Acido de xilosa	-	-	+	+	-	-	+	+
Acido de ribosa	V	-	+	-	ND	-	-	-
Símbolos: V, variable; ND, no determinado.


Mannheimia
Se recomienda aplicar los criterios recomendados por el fabricante. Se recomienda confirmar la identificación por métodos moleculares si el aislado es clínicamente relevante.
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica




Massilia
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica


Methylobacterium
Se recomienda aplicar los criterios recomendados por el fabricante, pero informar a nivel género
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica


Microbacterium
En la actualidad, se han descripto más de 80 especies dentro del género, pero sólo una minoría presenta importancia clínica. En la coloración de Gram se observan como
cocobacilos cortos o finos no ramificados. La actividad de catalasa y la motilidad son variables, y pueden ser fermentadores o presentar metabolismo oxidativo.Las patologías asociadas más frecuentes son bacteremia e infecciones de hueso, principalmente en pacientes oncológicos.
La identificación al nivel de especie resulta imposible mediante el análisis fenotípico, por lo que se requieren métodos moleculares (16S ARNr). Los aislamientos clínicos corresponden generalmente a M. oxydans, M. paraoxydans y M. foliorum.
Por ser bacilos Gram positivos del grupo pigmentado, MALDI-TOF puede identificar a nivel de género con valor de score >1,5.
En los BGP (especialmente el grupo de bacilos pigmentados) la No Identificación a nivel de especie no tiene mayor impacto. La importancia de la identificación radica en la ID de género.
El agregado de 1ul de ácido fórmico mejora la identificación
En base a la experiencia basada en escasos aislados, se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica



Micrococcus
El género fue redefinido manteniendo únicamente las especies M. luteus y M. lylae.
El hábitat principal de los Micrococcus y Dermacoccus es la piel de humanos y animales, y pueden actuar como patógenos oportunistas causando endocarditis, neumonía y sepsis en pacientes inmunocomprometidos.
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género
SCORE < 1,70 = no identifica







Moraxella
El género comprende alrededor de 20 especies, algunas de las cuales forman parte del microbioma del tracto respiratorio superior y otras son especies animales.
Se observan como cocos o cocobacilos que se disponen en pares o cadenas cortas y tienden a resistir la decoloración. Todas las especies son asacarolíticas y oxidasa positiva fuerte. M. catarrhalis y M. canis también dan la reacción de catalasa y DNAsa, y la mayoría de los aislamientos reducen nitratos a nitritos.
Existen pruebas fenotípicas que permiten diferenciar las especies clínicamente importantes del género .
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica

Ensayo	M. atlantae	M. canis	M. catarrhalis	M. lacunata	M. lincolnii	M.
nonliquefaciens	M. osloensis
Motilidad (flagelo)	-	-	-	-	-	-	-
Crecimiento en agar MacConkey	
+	
V	
-	
-	
-	
-	
V
Alcalinización de acetato	-	+	-	V	-	-	+
Susceptibilidad a desferrioxamina	
V	
+	
+	
+	
-	
+	
-
Acidificación de etilenglicol	-	+	-	V	-	-	+
Gelatinasa	-	-	-	+	-	-	-
Esterasa Tween 80	-	-	-	+	-	-	V
Tributirato esterasa	-	+	+	+	-	+	+
FAL	+	-	-	V	-	-	V
Phe desaminasa	-	-	-	-	-	-	-
PYR	+	-	-	-	-	-	-
Nitrato reductasa	-	V	+	+	-	+	V
Nitrito reductasa	-	V	+	-	-	-	-





Morganella
Debido a limitada experiencia en el género, se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica


Murdochiella
Cocos gram positivos anaerobios. Debido a limitada experiencia en el género, se recomienda aplicar los criterios recomendados por el fabricante, sin embargo en la cita de referencia los aislados de M. assacharolytica fueron identificados con score <1,7.
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica


Myroides
El género incluye dos especies, M. odoratimimus y M. odoratus, que pueden ser aislados a partir de muestras clínicas. Son bacilos inmóviles, con olor frutal similar a las especies de Alcaligenes faecalis. Presentan pigmento amarillo y crecen en la mayoría de los medios empleados comúnmente, con temperaturas óptimas de crecimiento entre 18 a 37°C. Son asacarolíticos, ureasa positiva, nitrato negativo y nitrito positivo. M. odoratus es susceptible a la desferrioxamina, mientras que M. odoratimimus es resistente.
La mayoría de los aislamientos provienen de orina, sangre e infecciones de oído y por lo general participan de infecciones polimicrobianas. Aunque las infecciones por Myroides son muy raras, se sabe que M. odoratimimus es 5 veces más frecuente que
M. odoratus. Las otras dos especies, aunque con mucha menos frecuencia, presentes en la base de datos, también han sido aisladas de especímenes humanos.
La mayoría de las cepas son resistentes a penicilinas, cefalosporinas, aminoglucósidos, aztreonam y carbapenemes.
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica



Neisseria
Se recomienda aplicar los criterios recomendados por el fabricante considerando las limitaciones descriptas a continuación
Identifica correctamente Neisseria gonorrhoeae
MALDI-TOF no logra la discriminación entre especies Neisseria cinerea y Neisseria flavescens/subflava. Además puede identificar Neisseria polysaccharea como Neisseria meningitidis.
MALDI-TOF identifica correctamente la especie Neisseria bacilliformis.
Debido a la estrecha similitud genética que existe entre las especies del género, se recomienda confirmar la identificación mediante Biología Molecular (16S ARNr, 23S ARNr).



Nocardia
Los fabricantes de ambas plataformas recomiendan la extracción en tubo para realizar la identificación de actinomicetales aeróbicos, por lo cual se recomienda realizar la 
técnica en caso de no obtener resultados por el método directo. La mayoría de las no identificaciones en MALDI-TOF ocurren por la dificultad para generar espectros de calidad que puedan ser comparados con los perfiles de referencia, y esto se debe fundamentalmente a la compleja composición química de su pared celular.
Se evaluaron los métodos de extracción recomendados por los fabricantes, y se evidenció que no son necesarios procedimientos complejos si se optimizan las condiciones al momento de la siembra, realizando simples pasos previos de extracción sobre colonias frescas.
Se obtienen resultados reproducibles al realizar la siembra por el método directo y con la disrupción en el pocillo con una gota de ácido fórmico. De no obtenerse el resultado esperado, se puede realizar la técnica de extracción EFAE Bruker.
Las especies N. cyriacigeorgica, Complejo N. farcinica, Complejo N. nova y Complejo
N. brasiliensis han sido validadas por FDA para amabas plataformas. Para el resto de las especies considerar las siguientes limitaciones:
Nocardia abscessus: informar complejo Nocardia abscessus (relacionada con Nocardia arthritidis, Nocardia asiatica y Nocardia exalbida). Si el score o % de identificación no cumple con los criterios del fabricante a pesar de la extracción en tubo, informar sólo género
Nocardia africana: Informar Complejo Nocardia nova.
Nocardia amikacinitolerans: Informar Nocardia sp., especie más relacionada N. amikacinotolerans
Nocardia anaemiae: Informar Nocardia sp.
Nocardia aobensis: Informar Nocardia sp.
Nocardia araoensis: Informar Nocardia sp, especies relacionadas Nocardia araoensis Nocardia beijingensis, Nocardia sputi, Nocardia sputorum, Nocardia niwae
Nocardia arizonensis: Informar Nocardia sp.
Nocardia arthritidis: informar Nocardia arthritidis (relacionada con Nocardia exalbida, N. asiática y N. abscessus).
Nocardia asiática: Informar N. asiática , relacionada con N. abscessus, N. arthitidis y N. exalbida
Nocardia asteroides: Informar especie
Nocardia blacklockiae: Informar Complejo Nocardia transvalensis. Nocardia brasiliensis: Informar complejo N. brasiliensis
Nocardia brevicatena: Informar Nocardia brevicatena/paucivorans. Nocardia carnea: Informar especie.
Nocardia cyriacigeorgica: Informar especie
Nocardia elegans: Informar Complejo Nocardia nova.
Nocardia exalbida: Informar N. exalbida, relacionada con N. arthitidis y N. gamkensis Nocardia farcinica: Informar Complejo N. farcinica
Nocardia ignorata: Informar Nocardia sp Nocardia inohanensis: Informar Nocardia sp
Nocardia kroppenstedtii: Informar Complejo Nocardia farcinica. Nocardia kruczakiae: Informar Complejo Nocardia nova.
Nocardia mexicana: Informar Nocardia sp
Nocardia niigatensis: Informar Nocardia sp Nocardia ninae: Informar Nocardia sp
Nocardia niwae: informar Nocardia sp. (especie más relacionada Nocardia niwae, difícil de diferenciar de Nocardia araoensis, Nocardia beijingensis, Nocardia sputi, Nocardia sputorum).
Nocardia nova: Informar Complejo Nocardia nova. Nocardia otitidiscaviarum: Informar especie.
Nocardia paucivorans: informar Nocardia brevicatena/paucivorans.
Nocardia pseudobrasiliensis: Si score >2 y discriminación >10% con otra especie: informar especie y controlar perfil de sensibilidad (cepas MDR)
Nocardia puris: Informar especie
Nocardia sienata: informar Nocardia testaceae/Nocardia sienata. Nocardia sputorum: Informar Nocardia sp
Nocardia testaceae: informar Nocardia testaceae/Nocardia sienata.
Nocardia terpenica: Informar Nocardia sp.
Nocardia thailandica: Informar Nocardia sp.
Nocardia transvalensis: Informar Complejo Nocardia transvalensis. Nocardia vermiculata: Informar Nocardia sp
Nocardia veterana: Informar Complejo Nocardia nova. Nocardia vinacea: Informar Nocardia sp.
Nocardia vulneris: Informar Complejo Nocardia brasiliensis. Nocardia wallacei: Informar Complejo Nocardia transvalensis.
La importancia de poder identificar ciertas especies es que se puede predecir el perfil de resistencia esperable a ciertos antibióticos.


•	Complejo N. brasiliensis: El perfil de sensibilidad a los antibióticos esperable para esta especie es: resistencia a imipenem, ciprofloxacina y claritromicina. Puede presentar resistencia a ceftriaxona. Un número significativo de aislamientos resistentes a Trimetoprima-sulfametoxazol ha sido documentado. En Argentina es la especie más asociada a infecciones de pile y partes blandas y no todas las cepas son resistentes a ciprofloxacina
•	Complejo N. farcinica: El perfil de sensibilidad a los antibióticos esperable para esta especie es: resistencia a ceftriaxona, tobramicina y claritromicina. Puede presentar resistencia a imipenem y minociclina.
•	N. cyriacigeorgica : El perfil de sensibilidad a los antibióticos esperable para esta especie es: resistencia a Amoxicilina-ácido clavulánico, ciprofloxacina y claritromicina. Puede presentar resistencia a minociclina.
•	Complejo N. nova : El perfil de sensibilidad a los antibióticos esperable para esta especie es: resistencia a Amoxicilina-ácido clavulánico, ciprofloxacina y tobramicina. Puede presentar resistencia a minociclina.
•	Complejo N. transvalensis : El perfil de sensibilidad a los antibióticos esperable para esta especie es: resistencia a amikacina, tobramicina y claritromicina. Puede presentar resistencia a Amoxicilina-ácido clavulánico, imipenem y minociclina. Un número significativo de aislamientos resistentes a Trimetoprima- sulfametoxazol ha sido documentado. Estas especies pueden presentar resistencia a las cuatro drogas de uso en la terapia empírica.
•	N. pseudobrasiliensis : El perfil de sensibilidad a los antibióticos esperable para esta especie es: resistencia a Amoxicilina-ácido clavulánico, imipenem, minociclina. Puede presentar resistencia a ceftriaxona y trimetoprima- sulfametoxazol. Estas especies pueden presentar resistencia a las cuatro drogas de uso en la terapia empírica.
•	N. otitidiscaviarum: El perfil de sensibilidad a los antibióticos esperable para esta especie es: resistencia a amoxicilina-ácido clavulánico, ceftriaxona, imipenem. Puede presentar resistencia a minociclina y claritromicina.


Ochrobactrum
El género Ochrobactrum incluye 18 especies, de las cuales O. anthropi (actualmente Brucella anthropi) y O. intermedia (actualmente Brucella intermedia) son las más asociadas a infecciones oportunistas en humanos.
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica.
Dado la estrecha relación filogenética entre O. anthropi y O. intermedia, muchas veces no se obtiene un 10% de divergencia entre las dos especies al revisar el Top Ten. Por ello, es conveniente incluir la sensibilidad a colistín como prueba adicional.
Pruebas adicionales recomendadas: colistín, tetraciclina, urea, 41C, NO3 (consultar Anexo).
Especies de Ochrobactrum spp. de importancia clínica.


Especie	
Informar	Nomenclatura actual/ Comentarios

O. anthropi	Ochrobactrum
grupo anthropi	Brucella anthropi
O. daejeonensis		Brucella daejeonensis
O. endophytica		Brucella endophytica
O. gallinifaecis		Brucella gallinifaecis
O. grignonensis		Brucella grignonensis
O. haematophilum		Brucella haematophila. Aislamiento humano
O. intermedium	Ochrobactrum
grupo intermedium	Brucella intermedia
O. pseudintermedium		Brucella pseudintermedia.
Aislamiento humano
O. pseudogrignonensis		Brucella pseudogrignonensis.
Aislamiento humano
Ochrobactrum sp.	Si score >2: informar Ochrobactrum sp.	
O. tritici	Ochrobactrum
grupo anthropi	Brucella tritici


El grupo Ochrobactrum anthropi agrupa las especies: O. anthropi, O. lupini, O. tritici y O. cytisi. Todas estas especies son sensibles al colistín, pero O.cytisi y O. lupini son resistentes a la Polimixina B 300 U.
El grupo Ochrobactrum intermediium incluye las especies: O. intermedium, O. pseudintermedium y O. pseudogrignonensis. Sin embargo, las mismas presentan diferencias con respecto a la sensibilidad a otros agentes antimicrobianos: Ochrobactrum intermedium presenta resistencia a colistín, netilmicina y desferrioxamina, pero es sensible a tetraciclina. En el caso de Ochrobactrum pseudointermedium, es resistente a colistín, tetraciclina y desferrioxamina, y sensible a netilmicina. Por último, Ochrobactrum pseudogrignonensis muestra resistencia a colistín, mientras que es sensible a tetraciclina, netilmicina y desferrioxamina.
Se sugiere que aunque todo el género Ochrobactrum fue transferido a Brucella, las especies de Ochrobactrum se deberían seguir informando como tal dado el alto impacto que tendría informar una especie de Ochrobactrum como Brucella.




Oligella
El género abarca dos especies: O. urethralis y O. ureolytica.
O. urethralis es un cocobacilo gram negativo, catalasa positiva, oxidasa positiva, inmóvil, que no oxida ni fermenta los hidratos de carbono, y no hidroliza gelatina ni esculina, además de ser ureasa negativa.
Estas pruebas la distinguen de O. ureolytica que es móvil y ureasa positiva rápida.
Las especies del género han sido descriptas como agentes causales de infección urinaria, vulvovaginitis, bacteremia y otras enfermedades sistémicas menos frecuentes, generalmente sobre pacientes inmunosuprimidos.
Mediante pruebas fenotípicas es necesario diferenciarla de Brevundimonas diminuta, especie estrechamente relacionada, de la cual se distinguen por ser cocobacilos inmóviles, sensibles al colistín.
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica




Olsenella
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica


Paenalcaligenes
Existe un único espectro de Paenalcaligenes hominis en la base de datos comercial. De las otras 2 especies descriptas: P. hermetiae y P. suwonensis, solo esta última ha sido descrita a partir de especímenes clínicos humanos.



Paenibacillus
Existe poca evidencia científica para evaluar la fiabilidad de la identificación a nivel de especie del género Paenibacillus. Dada su rareza en aislamientos clínicos, y debido a la limitada experiencia con aislamientos propios, se sugiere informar sólo a nivel de género según los criterios recomendados por el fabricante.
Es decir a partir de score >1,7 se informa Paenibacillus sp.





Pandoraea
Se recomienda aplicar los criterios recomendados por el fabricante. Sin embargo, debido a que ciertas especies aisladas principalmente de esputos de individuos con fibrosis
quística, recomendamos informar género y confrmar especie (si es relevante) mediante métodos moleculares



Pannonibacter
Se recomienda aplicar los criterios recomendados por el fabricante:

Pantoea
Limitada experiencia en el género, el cual además ha sufrido reclasificaciones taxonómicas de varias de sus especies.
Solo la especie P. agglomerans ha sido reclamada por amabas plataformas para ser validada en modo IVD
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica
Cambios taxonómicos
Especie	Nomenclatura actual / Comentarios
P. agglomerans	
P. allii	No aislamiento humano
P. ananatis	
P. anthophila	No aislamiento humano
P. brenneri	Aislamiento humano
P. calida	Mixta calida (Palmer et al. 2018)
P. citrea	Tatumella citrea (Brady et al. 2010).
No aislamiento humano
P. cypripedii	No aislamiento humano
P. conspicua	Aislamiento humano
P. deleyi	No aislamiento humano
P. dispersa	
P. eucrina	
P. gaviniae	Mixta gaviniae (Popp et al. 2010)
P. intestinalis	Mixta intestinalis (Prakash et al. 2015).
Aislamiento humano
P. punctata	Tatumella punctata (Brady et al. 2010).
No aislamiento humano
P. septica	
P. stewartii	Aislamiento humano
P. terrea	Tatumella terrea (Brady et al. 2010).
No aislamiento humano





Parabacteroides
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie. 
SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica



Paraburkholderia
Limitada experiencia en el género.
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica


Parvimonas
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica




Pasteurella
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69= identificación a nivel de género SCORE <1,5= no identifica.
Cambios taxonómicos Pasteurella spp.
Especie	Cambios taxonómicos/ Comentarios
P. gallinarum	Avibacterium gallinarum (Blackall et al. 2005).
Aislamiento humano
P. haemolytica	Mannheimia haemolytica (Angen et al. 1999)
P. mairii	No aislamiento humano
P. multocida	
P. oralis	Aislamiento humano
P.
pneumotropica	Rodentibacter pneumotropicus




Pediococcus
Se han aceptado valores de score para una identificación confiable en base a aislamientos de Pediococcus acidilactici y Pediococcus pentosaceus (especies más frecuentes en muestras clínicias):
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69= identificación a nivel de género SCORE <1,5= no identifica


Pruebas fenotípicas adicionales:
PYR: – LAP: +
NaCl: + Vancomicina: R Gas de Glucosa: – ADH: V




Peptococcus
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica





Peptoniphilus
El género Peptoniphilus incluye 27 especies, 17 validadas y 10 no validadas aún. (http://www.bacterio.net/peptoniphilus.html); 24 especies han sido aisladas de muestras clínicas humanas. Las especies P. harei y P. indolicus poseen MSP similares por lo que se deberán informar como una dupla.
	De acuerdo a los últimos cambios taxonómicos algunas especies son sinónimos; sin embargo, la forma correcta de informarlos es la siguiente:
P. lacydonensis en lugar de P. rhinitidis
P. tyrreliae en lugar de P. senegalensis
Se recomienda aplicar los criterios basados en publicaciones:
SCORE ≥ 1,8 = identificación a nivel de especie SCORE 1,79-1,60 = identificación a nivel de género SCORE < 1,60 = no identifica
Pruebas bioquímicas adicionales:
Pruebas fenotípicas para diferenciar las especies P. harei y P. indolicus.

Especie	
indol	
ureasa	
coagulasa	
FAL
P. indolicus	+	-	+	+
P. harei	v	-	-	-




Peptostreptococcus
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica



Plesiomonas
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica



Porphyromonas
Posee limitaciones en la identificación. En nuestra experiencia los scores obtenidos habitualmente son bajos y no es frecuente poder alcanzar la identificación. Por lo tanto, frente a un bacilo Gram negativo anaerobio estricto, realizar patrón de discos antibióticos:
-	Vancomicina (5ug) S
-	Colistina (10 ug) R
-	Kanamicina (1000 ug) R
-	Bilis 20% (oxgall) S


Si la identificación arroja una especie de Porphyromonas por MALDI-TOF con score
>1,7, informar a nivel de género (Porphyromonas sp.).
Las especies P. assacharolytica y P. uenonis no pueden diferenciarse por MALDI-TOF ni por 16S ARNr, y tiene limitada diferenciación con otros genes como hsp60, por lo tanto informar P. assacharolytica/uenonis.



Prevotella
Se recomienda aplicar los criterios basados en publicaciones:
SCORE ≥ 1,8= identificación a nivel de especie 
SCORE 1,79-1,60 = identificación a nivel de género SCORE < 1,60 = no identifica


Propionibacterium
Se recomienda seguir los criterios del fabricante. El género Propionibacterium atravesó cambios taxonómicos:
•	Propionibacterium	acidipropionici	es	Acidipropionibacterium acidipropionici.
•	Propionibacterium jensenii es Acidipropionibacterium jensenii.
•	Propionibacterium	microaerophilum	se	denomina	ahora
Acidipropionibacterium microaerophilum.
•	Propionibacterium	propionicum,	actualmente	se	conoce	como Pseudopropionibacterium propionicum.


Propionimicrobium
Existen 2 MSP en la base de datos de la única especie: P. lymphophilum.







Proteus
Se recomienda aplicar los criterios recomendados por el fabricante. Sin embargo, MALDI-TOF identifica correctamente Proteus mirabilis, pero no diferencia entre las especies Proteus vulgaris, Proteus penneri y Proteus hauseri; por lo tanto informar a nivel de grupo/complejo ó completar con pruebas fenotípicas para estas especies (ver Tabla a continuación).
Pruebas fenotípicas para diferenciar especies de Proteus spp.

Bacteria	TSI	Ureasa	LDC	IMVIC	ODC	Esc	Salicina	Trehalosa


P. vulgaris	Acido
+	
Desa- mina	

++-V	

-	+	+	-


P. penneri	Acido/ácido con o sin gas
Fondo negro	


+	

Desa- mina	


-+- -	


-	-	-	V
P. hauseri	Acido/ácido
sin gas	
+	Desa- mina	
++- -	
-	-	-	-
 


Pseudomonas
Se recomienda aplicar los criterios recomendados por el fabricante, pero dada la complejidad del género por las numerosas especies que contiene, existen consideraciones particulares:
Las especies Pseudomonas oryzihabitans, Pseudomonas aeruginosa, Pseudomonas stutzeri, Pseudomonas otitidis, Pseudomonas chlororaphis y Pseudomonas indica pueden ser bien identificadas a nivel de especie.
Pseudomonas salomonii no está en la base de datos (BD). El sistema MALDI-TOF la identifica como Pseudomonas antarctica/extremorientalis.
Pseudomonas cichorii/syringae: se recomienda confirmar por métodos moleculares.
Pseudomonas azotoformans no es identificada. Se sugiere realizar secuenciación para su confirmación.
Pseudomonas alcaliphila / oleovorans / pseudoalcaligenes: MALDI-TOF no logra discriminar estas especies; se recomienda el uso de genes gyrB para diferenciarlas.
Pseudomonas plecoglossicida, Pseudomonas monteilii, Pseudomonas mossellii, Pseudomonas putida y Pseudomonas fulva: estas especies se informan como> Pseudomonas grupo putida.
Pseudomonas lundensis, Pseudomonas vietnamiensis, Pseudomonas fluorescens, Pseudomonas libanensis, Pseudomonas koorensis y Pseudomonas synxantha: se informan como Pseudomonas grupo fluorescens.
Para estas especies no es posible alcanzar una diferenciación confiable mediante pruebas fenotípicas ni secuenciación del gen 16S ARNr. Se recomienda el uso de genes gyrB y rpoD para su confirmación.



Psychrobacter
Puede haber problemas en la identificación por ser cepas mucosas y pigmentadas. Suele arrojar valores bajos de score.
Las especies que se aíslan con mayor frecuencia son P. pulmonis, P. faecalis y P. inmobilis. 
Se recomienda informar solamente a nivel de género dado que dos de las principales especies de aislamiento humano, P. pulmonis y P. faecalis, no se encuentran en la base de datos. El sistema suele arrojar los aislados de estas especies como Psychrobacter sp.





Ralstonia
Posee gran similitud con el género Cupriavidus. Dentro del género, Ralstonia pickettii es la más conocida con respecto a enfermedad en el humano, causando bacteriemias, meningitis, endocarditis, osteomielitis. Ralstonia mannitolilytica fue recientemente descripta en un brote nosocomial y en un caso de meningitis recurrente; dicha especie junto a R. insidiosa afectan sobre todo a pacientes fibroquísticos. Basados en nuestra experiencia, Ralstonia pickettii es identificada correctamente. Sin embargo, R. mannitolilytica se confunde con R. pickettii, por lo tanto se sugiere informar todas las especies del género como Ralstonia sp.
Se recomienda informar a nivel de género con valor de score > 1,7.
R. solanaceum y R. syzygii forman parte del complejo Ralstonia solanaceum.



Raoultella
Se recomienda aplicar los criterios recomendados por el fabricante:
Sin embargo:
	Raoultella ornitinolytica: sólo informar como tal si la ODC es positiva y el Indol es positivo, de lo contrario informar como Klebsiella oxytoca, dado que aun con los 10 top ten para R. ornitinolytica puede tratarse de K. oxytoca.
	Raoultella planticola: informar como Klebsiella pneumoniae cuando el Indol es negativo, y como Klebsiella oxytoca cuando el Indol es positivo.
	Raoultella terrigena: la identificación a nivel de especie no es confiable.
Se recomienda consultar el Anexo para la diferenciación con Klebsiella spp. mediante pruebas fenotípicas.



Rhizobium
El género Rhizobium contiene numerosas especies, sin embargo existen muy pocos espectros depositados en la base de datos, excepto para R. radiobacter.
Se recomienda aplicar los criterios recomendados por el fabricante:



Rhodococcus
Se recomienda aplicar los criterios recomendados por el fabricante: Basados en nuestra experiencia y en datos publicados, R. equi puede identificarse correctamente a nivel especie con score >1,7.
Rhodococcus hoagii y Rhodococcus equi se consideran la misma especie.



Roseomonas
Estas especies son raramente aisladas de muestras clínicas (sangre, herida, absceso). Existen problemas en la identificación por MALDI-TOF por ser cepas mucosas y pigmentadas (rosado-coral), no mejorando con los métodos de extracción.
Puede no identificar algunas especies de Roseomonas, que además requieren secuenciación del gen 16S ARNr  para su confirmación: R. aestuarii /oryzae
/rhizosphareae /aerophila. De las 45 especies descriptas hasta la fecha (abril de 2021), solo las especies R. cervicalis, R. mucosa y R. gilardii han sido aisladas de infecciones humanas y están incluidas en la base de datos.
Se recomienda informar a nivel de género con valor de score > 1,7.
Todas las especies de Roseomonas hidrolizan fuertemente la urea pero no la esculina. De ser necesario, se sugiere agregar las siguientes pruebas fenotípicas:
Pruebas fenotípicas diferenciales para especies de Roseomonas spp.




Especie	Oxidasa	PYR	Acido de Arabinosa	Manitol	Fructosa	Glucosa	NO3	Desferroxiamina
R. cervicalis	+	-	+	-	V	-	-	-
R. gilardii	+	+	+	+	V	-	-	-
R. mucosa	-	+	+	+	+	+	-	-
R. genomospecie 4	+	-	-	-	+	+	+	+
R. genomospecie 5	+	-	-	-	+	-	-	-
Símbolos: V, variable.







Rothia
El género se presenta como cocos gram positivos (representado por R. mucilaginosa), aunque también pueden aparecer como cocobacilos o bacilos del tipo corineiforme, con tendencia a la ramificación rudimentaria (representado por R. dentocariosa)
Las especies del género clínicamente relevantes son R. mucilaginosa (originalmente
Stomatococcus mucilaginosus), R. dentocariosa y R. aeria.
Forman parte de la flora normal de cavidad bucal y orofaringe en personas sanas, aunque las lesiones periodontales pueden constituir la vía para el desarrollo de una bacteriemia u otra enfermedad sistémica.
R. aeria ha sido asociada a endocarditis y sepsis; mientras que Rothia mucilaginosa ha causado meningitis y septicemia sobre todo en niños con enfermedades hematológicas.
Las colonias de R. dentocariosa suelen ser blanquecinas (o muy raramente negro grisáceas, adherentes), suaves o rugosas, o con forma de “rueda de carro”, y crecen mejor en atmósfera de CO2. Esta especie es catalasa variable, inmóvil, reduce nitratos, hidroliza esculina, es ureasa negativa, y fermenta glucosa, maltosa y sacarosa, pero da negativa la lactosa, xilosa y manitol.
El API Coryne identifica correctamente la especie representativa del género: Rothia dentocariosa (PAL y Βgur positivas).
Las pruebas fenotípicas convencionales no logran la diferenciación entre R. aeria y R. dentocariosa, pero MALDI-TOF sí logra la identificación al nivel de especie.
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica




Ruminococcus
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica



Salmonella
No existe suficiente experiencia de los LNR en el género. Informar solo género
Se sugiere informar como Salmonella enterica subsp. enterica (excluída Salmonella Typhi), una vez que esta última fue descartada por pruebas bioquímicas, dado que S. Typhi no esta incluida en la base de datos
Diferenciación bioquímica entre Salmonella enterica serovar Typhi y Salmonella enterica
(no Typhi)

Bacteria	Levine	TSI	LDC	Ureasa	ODC	IMVIC

S. enterica
Serovar
Salmonella Typhi	
Sin cambio	
Alcalino ácido
anillo negro sin gas	

+	

-	-	

-+--
S. enterica (excluído Serovar
Salmonella Typhi)	

Sin cambio	
Alcalino fondo negro gas	

+	

-	+	

-+-+

Se efectúa la confirmación serológica empleando los antisueros correspondientes.
Salmonella Typhi: presenta el Ag Vi






Serratia
No existe suficiente experiencia de los LNR en el género.Se recomienda aplicar los criterios recomendados por el fabricante, haciendo la salvedad que cuando el sistema arroja S. ureilytica, debe confirmarse la identificación dado que debido a la similtud entre los espectros de ambas especies, el sistema confunde a S. marcescens con S. ureilytica
Pruebas de diferenciación fenotípica de las especies S. ureilytica y S. marcescens.

Especie	Urea	Caseina	ADH	Adonitol	Xilosa	Melibiosa
S. ureilytica	+	+	+	+	+	+
S. marcescens	-	-	-	-	-	-




Shewanella
Se recomienda informar a nivel de género a partir de score > 1,7.
Shewanella spp. es el único género de los BGNNF que produce ácido sulfhídrico en el TSI. Shewanella algae puede ser erróneamente identificada como Shewanella putrefaciens. Aclaración: S. algae, representa la mayoría de los aislamientos humanos y S. putrefaciens representa la mayoría de los aislamientos no humanos.
Pruebas bioquímicas para diferenciar especies de Shewanella spp.

Especie	
Pigmento	Desarrollo en NaCl 6.5%	
OF
Fructosa	
OF
Sacarosa	
OF
Maltosa	
Desarrollo en SS
S. algae	Tostado	+	-	-	-	+
S. putrefaciens	Tostado	-	V	+	+	-
Símbolos: V, variable.
La completa diferenciación de las especies del género se realiza mediante secuenciación de dianas genéticas específicas tales como: 16S ARNr, 16S-23S, 23S ARNr, gyrB, rpoB, recA.



Slackia
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica






Shigella
No distingue de E. coli. Limitación del método
 






Sphingobacterium
Hasta el momento y de acuerdo a nuestra experiencia, se recomienda aplicar los criterios recomendados por el fabricante:
La especie que se aisla con mayor frecuencia en la clínica es, sin duda, S. multivorum.







Sphingomonas
Hasta el momento se sugiere informar a nivel de género a partir de score > 1,7.
S. paucimobilis es un bacilo gram negativo polimórfico, aerobio estricto, oxidasa positiva débil y catalasa positiva. Las colonias crecen en agar sangre pero no en agar MacConkey, y producen pigmento amarillo. Aunque posee un único flagelo polar, un bajo porcentaje de células son activamente móviles, y la motilidad puede ser difícil de demostrar en el laboratorio (de ahí el nombre paucimobilis). Sphingomonas puede ser erróneamente identificado por sistemas convencionales de identificación, pero suele ser correctamente identificado por MALDI-TOF.
Se ha documentado que los miembros del género Sphingomonas incluyen más de un grupo filogenético, cada uno de los cuales representa un género diferente: Sphingobium, Novosphingobium, Sphingopyxis y Sphingomonas (Takeuchi M et al., 2001). Por el momento se recomienda informar Sphingomonas spp. en el laboratorio clínico dada la diversidad genética del género.





Staphylococcus
Se sugiere utilizar un método de extracción directa (con 1 µl de ácido fórmico al 70 %) y considerar los siguientes valores de corte:
SCORE ≥1,7 = identificación correcta a nivel de especie SCORE 1,5-1,7 = identificación correcta a nivel de género SCORE <1,5 = identificación no confiable
Se debe considerar lo siguiente al informar los resultados:
Confiabilidad de la identificación: La precisión de la identificación por MALDI-TOF varía según la especie. Algunas especies, como S. aureus y S. epidermidis, se identifican con alta confiabilidad. Sin embargo, para otras especies, como S. arlettae la identificación puede no ser confiable, independientemente del score obtenido. En estos casos, se debe considerar realizar pruebas adicionales para confirmar la identificación.
Nivel de discriminación: MALDI-TOF puede discriminar entre especies y subespecies de Staphylococcus. Es importante informar el nivel de identificación alcanzado. Por ejemplo, si se identifica S. capitis a nivel de subespecie (S. capitis ssp capitis o S. capitis ssp ureolyticus), se debe especificar la subespecie.
Especies con identificación problemática: Algunas especies, como S. pseudointermedius, pueden generar identificaciones incorrectas por MALDI- TOF.En estos casos, se recomienda informar solo el grupo al que pertenece la especie (en este caso, Grupo S. intermedius (SIG)).


Información adicional relevante: Se debe incluir información adicional que pueda ser relevante para el clínico, como la resistencia a la novobiocina, el origen de la muestra y la asociación con enfermedades específicas.
Ejemplos de cómo informar las identificaciones:
S. aureus identificado por MALDI-TOF.
S. capitis ssp ureolyticus identificado por MALDI-TOF.
Staphylococcus spp. del Grupo S. intermedius (SIG) (identificación por MALDI- TOF no confiable a nivel de especie).
S. saprophyticus ssp saprophyticus, novobiocina resistente, aislado de muestra de orina.
Especies de Staphylococcus spp. incluidas en la Base de Datos Biotyper software 3.1 (Bruker Daltonics).

ID por MALDI TOF	

INFORMAR	

Observaciones
S. argenteus	

SI	Maldi-tof lo diferencia de S. aureus (sin embargo se recomienda reportar como complejo S. aureus)
S. chromogenes	

SI	Aislado de animales de pezuñas hendidas o partidas (artiodáctilos): ovejas, cabras, venados, camellos, ganado vacuno y cerdos). Produce mastitis bovina.
S. arlettae	SI
(habitualmen te se obtiene ID no confiable, independient emente del score usado)	Especie de ECN Novobiocina R aislada de animales
S. aureus	SI	
S. auricularis	SI	Identificación solo a nivel de género
S. capitis ssp
capitis	SI	Discrimina a nivel de subespecie
S. capitis ssp
ureolyticus	SI	Discrimina a nivel de subespecie
S. caprae	SI	Especie aislada de leche de cabra. Patógeno emergente en infecciones humanas (osteoarticulares, endocarditis, etc)
S. carnosus ssp
carnosus	SI (por lo menos a	Los distintos trabajos no especifican si discrimina a nivel de subespecie nivel de especie)	
S. carnosus ssp
utilis	SI (por lo menos a nivel de especie)	Los distintos trabajos no especifican si discrimina a nivel de subespecie
S. cohnii ssp
cohnii	SI	Discrimina a nivel de subespecie.
Especie de ECN Novobiocina R
S. cohnii ssp
urealyticus	SI	Discrimina a nivel de subespecie.
Especie de ECN Novobiocina R
S. condimenti	SI	Aislado de muestras de alimentos (salsa de soja). También aislado de infecciones humanas (bacteriemia asociada a catéter)
S. delphini	SI	Especie coagulasa positiva, aislada de delfines y de caballos. Forma parte del grupo S. intermedius (SIG) junto con S. intermedius y S. pseudointermedius
S. epidermidis	SI	
S. equorum	SI	Especie de ECN Novobiocina R aislada de animales
S. felis	SI	Especie aislada de muestras clínicas de gatos
S. fleurettii	Da scores bajos	Usando >2 (para especie) solo discrimina a nivel de género.
Usando >1,7 (para especie) ID incorrecta como S. sciuri. Especie Novobiocina Resistente
S. gallinarum	ND	Especie de ECN Novobiocina R aislado de animales
S. haemolyticus	SI	
S. hominis ssp
hominis	SI	Discrimina a nivel de subespecie
S. hominis ssp
novobiosepticus	SI	Discrimina a nivel de subespecie.
Novobiocina R
S. hyicus	SI	Especie coagulasa variable aislada de cerdos
S. intermedius	SI	ID correcta a nivel de especie, independientemente del score usado. Especie coagulasa positiva, aislada de palomas. Forma parte del grupo S. intermedius (SIG) junto con S. delphini y
S. pseudointermedius
S. kloosii	SI	Especie de ECN Novobiocina R aislada de animales
S. lentus	SI	Ex Staphylococcus sciuri ssp lentus. Especie de ECN Novobiocina R
S. lugdunensis	SI	
S. lutrae	SI	Especie coagulasa positiva aislada de nutrias
S. microti	ND	Especie aislada de Microtus arvalis (una especie de roedor de la familia Cricetidae ampliamente distribuido por Europa y algunas zonas de Asia). Especie Novobiocina R
S. muscae	SI	Especie aislada de moscas
S. nepalensis	SI	Especie aislada de cabras del Himalaya.Novobiocina R
S. pasteuri	SI	Especie aislada de muestras humanas, animales y alimentos. Su nombre es en honor al microbiólogo francés Louis Pasteur por su contribución en 1878 al reconocimiento de estafilococos como agentes patógenos y también al instituto de investigación Instituto Pasteur, París, Francia, donde se caracterizó la nueva especie
S. petrasii	SI	Especie aislada de muestras humanas (infecciones óticas, PPB, etc)
S. pettenkoferi	SI	Especie de ECN aislada de muestras humanas
S.piscifermentans	SI	Especie de ECN aislada de pescado fermentado en Tailandia
S.pseudointermedi us	NO (Solo informar como grupo Staphylococc us intermedius): SIG	ID incorrecta como S. intermedius independientemente del score usado.
Especie coagulasa positiva, aislada de perros y gatos domésticos. Forma parte del grupo S. intermedius (SIG) junto con
S. intermedius y S. delphini
S.saccharolyticus	SI	Especie anaerobia de Staphylococcus.
Anteriormente clasificado como
Peptococcus saccharolyticus
S. saprophyticus
ssp bovis	SI (por lo menos a nivel de especie)	Los trabajos no especifican si discrimina a nivel de subsp. Es de aislamiento animal (fosas nasales bovinas).
Novobiocina R
S. saprophyticus ssp saprophyticus	SI (por lo menos a nivel de especie)	Los trabajos no especifican si discrimina a nivel de subespecie. Novobiocina R
S. schleiferi ssp
coagulans	SI	Especie coagulasa positiva
S. schleiferi ssp
schleiferi	SI	
S. schweitzeri	SI	Se recomienda reportar como complejo
S. aureus
S. sciuri ssp
carnaticus	SI (por lo menos a nivel de especie)	Novobiocina R.
Los trabajos no especifican si discrimina a nivel de subsp.
S. sciuri ssp
rodentium	SI (por lo menos a nivel de especie)	Novobiocina R.
Los trabajos no especifican si discrimina a nivel de subsp.
S. sciuri ssp sciuri	SI	Novobiocina R.
S. simiae	SI	Especie aislada de monos ardilla de América del Sur
S. simulans	SI	Denominado como tal por su similitud con ciertas especies de estafilococos coagulasa positiva (incluyendo S. aureus)
S. succinus ssp
casei	SI (por lo menos a nivel de especie)	Los trabajos no especifican si discrimina a nivel de subsp. Especie aislada de la superficie del queso madurado.
Novobiocina R.
S. succinus ssp
succinus	SI (por lo menos a nivel de especie)	Los trabajos no especifican si discrimina a nivel de subsp. Especie Novobiocina R, aislada de ámbar dominicano.
S. vitulinus	SI	Nombre actual de S. pulvereri.
Novobiocina R.
S. warneri	SI	Su nombre es en honor a Arthur Warner, quien originalmente aisló este microorganismo.
S. xylosus	SI	Novobiocina R
Especies de Staphylococcus spp. NO incluidas en la Base de Datos de Bruker:
1)	 De aislamiento animal o del ambiente:
Staphylococcus agnetis (Taponen et al. 2012, sp. nov.). Especie coagulasa variable aislada de leche bovina.
Staphylococcus argensis (Heβ and Gallert 2015, sp. nov.)
Staphylococcus caseolyticus (ex Evans 1916) Schleifer et al. 1982, nom. rev., comb. nov. (Actualmente: Macrococcus caseolyticus subsp. caseolyticus y Macrococcus caseolyticus subsp. hominis).
Staphylococcus edaphicus (Pantůček et al. 2018, sp. nov.)
Staphylococcus rostri (Riesen and Perreten 2010, sp. nov.). Aislado de las narinas de cerdos sanos (Riesen et al., 2010).
Staphylococcus schweitzeri (Tong et al. 2015, sp. nov.). Forma parte del complejo Staphylococcus aureus pero se lo ha aislado de primates NO humanos (Tong et al., 2015)
Staphylococcus stepanovicii (Hauschild et al. 2012, sp. nov.). Especie Novobiocina R y Oxidasa positiva aislada de ciertos pequeños mamíferos salvajes (Hauschild et al., 2010).


2)	 De aislamiento humano:
Staphylococcus jettensis (De Bel et al. 2013, sp. nov.).
Staphylococcus massiliensis (Al Masalma et al. 2010, sp. nov.) Aislado de absceso de cerebro (Al Masalma et al., 2010).
Staphylococcus petrasii (Pantůček et al. 2013, sp. nov.) Aislado de infecciones óticas (Pantucek et al., 2013).



 



Stenotrophomonas
S. maltophila aunque no es típicamente patógena para personas sanas, es un patógeno oportunista bien conocido para el hombre. Está dentro de las causas más comunes de infección de herida debido a traumas que involucran maquinaria agrícola. Es también un importante patógeno nosocomial asociado a alta morbilidad y mortalidad, particularmente en pacientes debilitados o inmunocomprometidos, y pacientes que requieren ventilación artificial en UCI. La incidencia de infecciones humanas ha aumentado en los últimos años, y una variedad de sindromes clínicos han sido descriptos, incluyendo bacteriemia, neumonía, infección del tracto urinario, infección ocular, endocarditis, meningitis, infección de herida y partes blandas, mastoiditis, epididimitis, colangitis, osteocondritis, bursitis y peritonitis. La septicemia puede estar acompañada por ectima gangrenosa, una lesión de piel más comúnmente asociada con
P. aeruginosa y Vibrio spp. La incidencia de infección del tracto respiratorio por S. maltophila en personas con fibrosis quística también parece ir en aumento.
No se logra discriminación entre las especies del género, que suelen arrojar bajos valores de score.Informar a nivel de género con valore de score > 1,7.
Por otra parte, MALDI-TOF puede identificar erróneamente Stenotrophomonas maltophilia como especies de Pseudomonas, por lo que ante la sospecha, se sugiere agregar las siguientes pruebas fenotípicas características del género:
Oxidación de glucosa y maltosa (intensa), DNAsa, Lisina Decarboxilasa, movilidad: todas arrojan resultado positivo.






Streptococcus
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica.
MALDI-TOF puede identificar erróneamente Streptococcus mitis como Streptococcus pneumoniae y viceversa. Se debe realizar la prueba de solubilidad en bilis y optoquina (en O2 y con 5% de CO2).
No diferencia S. pneumoniae de S. pseudopneumoniae: realizar la prueba de la Optoquina en O2 y CO2.
No diferencia entre especies del grupo mitis (menos del 10% de divergencia): informar como grupo mitis.
MALDI-TOF identifica correctamente las especies de Streptococcus pyogenes.
Pruebas adicionales:Streptococcus Beta hemolíticos: Bacitracina, PYR, CAMP, serología, VP, Sorbitol, Trehalosa.
Streptococcus grupo viridans. Se deben informar a nivel de GRUPO, los siguientes:
Streptococcus grupo
mitis	S. mitis
	S. sanguinis
	S. parasanguinis
	S. gordonii
	S. cristatus
	S. oralis
	S. infantis
	S. peroris
	S. australis
	S.oligofermentans
	S. massiliensis
	S. sinensis
	S. orisratti
	S. pseudopneumoniae
	S. pneumoniae
	S. tigurinus
	S. lactarius 
Streptococcus grupo
anginosus	S. anginosus
	S. constellatus
	S. intermedius
Streptococcus grupo
salivarius	S. salivarius
	S. vestibularis
	S. thermophilus
Streptococcus grupo
mutans	S. mutans
	S. sobrinus
	S. cricetti
	S. ratti
	S. downei
Se pueden informar al nivel de ESPECIE los siguientes:Streptococcus grupo
bovis	S. lutetiensis	Informar a nivel de especie
	S.equinus	Confirmar con genes rpoB y sodA
	S. gallolyticus ss gallolyticus	Informar a nivel de especie
	S. gallolyticus ss pasteurianus	Informar a nivel de especie
	S. infantarius	Informar a nivel de grupo





Streptomyces
La taxonomía del género, compuesto por más de 525 especies y subespecies, continúa siendo problemática. Muchas de estas especies han sido patentadas debido a que los productos que sintetizan son usados con fines comerciales.
Hasta el momento, la secuenciacion de los genes 16S ARNr y gen secA permite distinguir entre las especies más frecuentes de Nocardia y Gordonia / Streptomyces / Tsukamurella. Puede causar enfermedad en pacientes inmunocomprometidos y raramente en individuos sanos, siendo la más común el micetoma cuyo agente etiológico suele ser Streptomyces somaliensis.Sin embargo, esta especie no se encuentra, a la fecha, en la base de datos del sistema.
Existen algunos reportes que implican otras especies del género como patógenos oportunistas. Debido al gran número de especies descriptas de Streptomyces, y la falta de información sobre el impacto clínico de muchas de ellas, la identificación a nivel de género es probablemente suficiente en la mayoría de los casos.
En un estudio sobre la susceptibilidad de 92 especies de Streptomyces de aislados clínicos, 100% fueron sensibles a amikacina y linezolid, 77% a minociclina, 67% a imipenem y 51% a claritromicina y amoxicilina-ácido clavulánico.
Micobacterias, Nocardia y Actinomicetes aeróbicos constituyen un desafío diagnóstico debido a la compleja pared celular que poseen, por lo que pueden requerir un procesamiento especial previo al análisis por MALDI-TOF para obtener resultados más precisos.
En base a la experiencia de los laboratorios de referencia, hay necesidad de aumentar los perfiles representativos para este grupo de microorganismos en la base de datos comercial (Loucif et al. 2014). Por otra parte, debido a las características de la pared celular, suelen obtenerse no identificaciones o bajos valores de score mediante MALDI- TOF, por lo que se recomienda seguir los siguientes pasos:
a)	sembrar la muestra por el método directo,
b)	sembrar la muestra con el posterior agregado de 1ul de ácido fórmico al pocillo,
c)	cubrir la muestra con 2ul de matriz HCCA,
d)	sembrar la muestra por el método directo, pero partiendo de placas envejecidas, sembradas con varios días de anterioridad,
e)	de no obtenerse el resultado esperado, se deberá realizar el proceso de extracción con etanol y ácido fórmico. También se puede ensayar el proceso de extracción con perlas para Actinomicetales, recomendado por el fabricante.
Se recomienda informar a nivel de género con valor de score ≥ 1,7.



Terribacillus
Las especies del género son T. aidingensis, T. goriensis, T. halophilus, T. saccharophilus. Las mismas suelen ser ambientales y no revisten importancia clínica.
En nuestra experiencia, Terribacillus goriensis / saccharophilus (identificado mediante Biología Molecular), puede ser identificado por MALDI-TOF como Brevibacillus brevis con valor de score > 2.
No existen perfiles proteicos de referencia para Terribacillus spp. en las bases de datos comerciales actuales.





Tetragenococcus solitarius
Se desconoce su rol para causar infección en el humano.
Existe un único MSP en la Base de Datos comercial Maldi Biotyper 3.1.




Tissierella
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica
 



Trueperella
El género comprende cinco especies, de las cuales Trueperella pyogenes y Trueperella bernardiae (Arcanobacterium bernardiae hasta el año 2011) pueden ser aisladas a partir de especímenes clínicos, más frecuentemente infecciones de piel y abscesos. Las especies de Arcanobacterium son CAMP / CAMP reverso positivo, mientras que Trueperella arroja resultado negativo. T. pyogenes es patógeno veterinario y raramente causa infección en el humano. Es la única especie de Arcanobacteria / Trueperella de importancia clínica, con actividad Beta Glucuronidasa y Fermentación de Xilosa positivas. T. bernardiae reduce Maltosa y Glucosa.
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69 = identificación a nivel de género SCORE <1,5 = no identifica




Tsukamurella
La composición química característica de su pared celular la separa del resto de los Actinomicetales. La especie tipo del género es Tsukamurella paurometabola (originalmente Corynebacterium paurometabolum), seguida de doce especies. Las infecciones suelen estar asociadas a cuerpos extraños, como catéteres intravenosos.
T. tyrosinosolvens ha estado implicada en casos de queratitis, bacteriemia, infección asociada a catéter. Liu y colaboradores, reportaron los datos de sensibilidad para T. tyrosinosolvens, T. spumae y T. pulmonis usando un corto período de incubación (las normas del CLSI siguen indicando la lectura a las 48 horas), resultando las tres especies sensibles a Amoxicilina-Ácido clavulánico, Ciprofloxacina y Linezolid. T. tyrosinosolvens y T. spumae fueron sensibles a sulfametoxazol, y T. pulmonis fue reportada como resistente.
En base a la experiencia de los laboratorios de referencia, hay necesidad de aumentar los perfiles representativos para este grupo de microorganismos en la base de datos comercial. Por otra parte, debido a las características de la pared celular, suelen obtenerse no identificaciones o bajos valores de score mediante MALDI-TOF, por lo que se recomienda seguir los siguientes pasos:
a)	sembrar la muestra por el método directo,
b)	sembrar la muestra con el posterior agregado de 1ul de ácido fórmico al pocillo,
c)	cubrir la muestra con 2ul de matriz HCCA,
d)	sembrar la muestra por el método directo, pero partiendo de placas envejecidas, sembradas con varios días de anterioridad,
e)	de no obtenerse el resultado esperado, se deberá realizar el proceso de extracción con etanol y ácido fórmico. También se puede ensayar el proceso de extracción con perlas para Actinomicetales, recomendado por el fabricante.
Se recomienda informar a nivel de género con valor de score ≥ 1,7.




Ureaplasma
No existe perfil de referencia en la base de datos comercial.
 



Vagococcus
Hasta la actualidad, ha sido muy escaso el número de Vagococcus aislados a partir de especímenes clínicos (hemocultivo, herida, líquido peritoneal). Las dificultades que se dan en la identificación son debidas a su baja frecuencia. Se recomienda informar la identificación a nivel de género con valor de score >1,5.


Varibaculum
De las dos especies que se aíslan en clínica (V. anthropi y V. cambriense), solo existe un único MSP de esta última en la Base de Datos comercial Maldi Biotyper 3.1.





Veillonella
El género Veillonella contiene 15 especies, de las cuales V. atypica, V. denticariosi, V. dispar, V. infantium, V. nakazawae, V. parvula, V. rogosae y V. tobetsuensis han sido aisladas de la cavidad oral humana.
Los factores de riesgo para una infección por Veillonella incluyen periodontitis, inmunodeficiencia, uso de drogas intravenosas y parto prematuro.Son agentes etiológicos de infecciones severas tales como meningitis, osteomielitis, infección protésica, bacteremia y endocarditis; aunque se desconocen los mecanismos de virulencia de estos microorganismos. Muestra resistencia a tetraciclina, eritromicina, gentamicina y kanamicina, y es sensible a penicilina G, cefalotina y clindamicina.
Debido a la escasa experiencia de los laboratorios de referencia, y en base a la bibliografía que se adjunta al pie, donde se evidencia la limitación de la espectrometría de masas para alcanzar identificaciones a nivel de especie, se recomienda informar los resultados de Veillonella spp. sólo al nivel de género.SCORE > 1,7 = identificación a nivel de género


Vibrio
Los miembros de la familia Vibrionaceae pueden ser causantes de una gran variedad de enfermedades intestinales y extraintestinales en el humano; las mismas incluyen diarrea, celulitis, fascitis necrotizante, septicemia e infecciones de ojo y oído.
Las especies con significancia clínica se detallan a continuación:
Especie	Manifestación clínica
V. cholerae	Se divide en más de 200 serogrupos, de los cuales únicamente el serogrupo O1 y O139 son los responsables del cólera epidémico y pandémico.
V. mimicus	Diarrea.
V. parahaemolyticus	Infección intestinal; asociado al consumo de pescado crudo.
V. vulnificus	Septicemia, infección de heridas y de oído.
V. fluvialis	Gastroenteritis, bacteremia.
V. furnissii	Diarrea.
Se recomienda informar según recomendación del fabricante:
SCORE ≥ 2,00 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,70 = no identifica.
Vibrio cholerae no se encuentra representado en la base de datos comercial por considerarse un agente de bioterrorismo. En el caso de los agentes que requieren Nivel 3 de Bioseguridad, es importante el método de elección para preparar las muestras que serán procesadas en MALDI-TOF, ya que debe asegurar tanto la inactivación del microorganismo como la calidad óptima del espectro generado. En base a la bibliografía, se recomienda el método de extracción con etanol / ácido fórmico / acetonitrilo.



Wautersiella falsenii
Este género presenta características fenotípicas similares a las de miembros de los géneros Chryseobacterium y Empedobacter (indol positivo). Ha sido aislado de muestras clínicas.
Se recomienda aplicar los criterios recomendados por el fabricante:
SCORE ≥ 2,0 = identificación a nivel de especie SCORE 1,7-1,99 = identificación a nivel de género SCORE < 1,7 = no identifica




Weeksella
El hábitat natural de los BGNNF, oxidasa e indol positivos, suele ser el suelo, las plantas, el agua, incluyendo las del ámbito hospitalario. La especie clínicamente relevante es Weeksella virosa, aislada con frecuencia del tracto urogenital. Se recomienda informar según recomendación del fabricante:



  




Weissella
Es miembro del grupo de cocos Gram positivos catalasa negativos, PYR negativo, vancomicina resistente, junto a Leuconostoc y Pediococcus. La especie más frecuente del género es Weissella confusa, agente causal de bacteremia y endocarditis.
Se han aceptado valores de score para una identificación confiable:
SCORE >1,7 = identificación a nivel de especie SCORE 1,5-1,69= identificación a nivel de género SCORE <1,5= no identifica
En el caso de Weissella paramesenteroides y Weissella confusa pueden ocurrir fallas en la identificación (No Identificación). Para confirmar la identificación a nivel de especie se recomienda realizar la secuenciación del gen 16S ARNr o sodA.
Características fenotípicas de especies de Weissella spp.
Ensayo	
W. confusa	
W. cibaria	
W. viridescens	
W.
paramesenteroides*
Acido de Arabinosa	-	+	-	d
Acido de Galactosa	+	-	-	+
Acido de Ribosa	+	-	-	d
Acido de Sacarosa	+	+	d	+
ADH	+	+	-	-
Hidrólisis de esculina	+	+	-	v
*W. paramesenteroides no ha sido aislada en muestras clínicas.







Wohlfahrtiimonas
W. chitiniclastica es un bacilo gram negativo no fermentador cuyo hábitat natural es el intestino de larvas de algunas moscas parásitas, clasificado en el pasado como "bacilo grupo 1 de Gilardi". Puede estar asociado con miasis humana, a veces resultando en bacteriemia o sepsis fulminante. Se lo aisla además de la sangre y de heridas infectadas.
Se recomienda aplicar los criterios recomendados por el fabricante:



Yersinia
No existe experiencia propia en el género.
Yersinia representa un grupo de microorganismos clínicamente importantes pero poco frecuentes en los aislamientos clínicos. Las especies del género que causan enfermedad en el humano son Yersinia pseudotuberculosis, Yersinia enterocolitica y Yersinia pestis. Las infecciones causadas por Y. pseudotuberculosis y Y. enterocolitica ocurren luego de la ingestión de comida o agua contaminada, y se manifiestan primariamente como una gastroenteritis; mientras que Yersinia pestis (agente etiológico de la plaga), es transmitida al humano a través de la mordida de una pulga infectada y resulta en una grave enfermedad con un alto grado de mortalidad. Nuevas especies descriptas recientemente: Yersinia artesiana , Yersinia proxima, Yersinia alsatica, Yersina vastinensis, Yersinia thracica y Yersinia occitanica han sido aisladas de heces humanas (Le Guern AS et al., 2020). Asimismo Y. canariae (descripta por Nguyen SV et al.) ha sido asociada a yersiniosis humana.
Dentro del género, existen otras 26 especies ambientales y no patogénicas para el hombre.
Yersinia pestis es considerada un agente de bioterrorismo, que debe ser manipulado en cabinas de bioseguridad al menos de Nivel 2 (BSL-2, BSL-3), por lo que no se encuentra representada en la base de datos comercial del equipo.
El uso de MALDI-TOF para este grupo de microrganismos ha sido evaluado de diversas formas; en 2010 Lasch y colaboradores desarrollaron una base de datos de referencia mediante la cual se logró la detección de picos característicos a nivel de género y especie, y picos biomarcadores entre Y. pestis / Y. pseudotuberculosis, organismos genéticamente muy relacionados.
Por otra parte, Ayyadurai y colaboradores, lograron la diferenciación de especies ambientales y patogénas de Yersinia, y la detección de biotipos de Y. pestis mediante la creación de una base de datos representando 12 especies y 3 biotipos de Yersinia pestis.
En base a la bibliografía, se podría considerar a MALDI-TOF como un método robusto y confiable para la identificación de especies de Yersinia, el cual puede aportar además información epidemiológica al detectar biotipos de Yersinia pestis. Sin embargo, el aspecto más importante a tener en cuenta es el protocolo de inactivación de los microorganismos a ensayar; ya que además de la bioseguridad, debe tener una mínima influencia sobre el espectro generado. La metodología más utilizada en la actualidad es la extracción con TFA.

    """,
    tools='code_execution',
)

# Iniciar la sesión de chat
chat_session = model.start_chat(history=[])

print("Bot: ¡Bienvenido! Soy MALDI_Bot, su asistente en espectrometría de masas. Estoy aquí para ayudarle a interpretar resultados de MALDI-TOF y guiarle en la identificación precisa de bacterias. Si tiene alguna pregunta específica o necesita orientación sobre un paso del proceso, no dude en preguntar.")
print()

# Página principal
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint para procesar mensajes
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['message']
    response = model.start_chat(history=[]).send_message(user_input)
    return jsonify({'response': response.text})

if __name__ == '__main__':
    app.run(debug=True)