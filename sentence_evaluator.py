

##in case we need to translate the sentences.
def translate_sentence(raw):
    sentence = TextBlob(raw)
    sentence_en = sentence.translate(to='en')
    return sentence_en

##determinar si un texto es muy dificil de leer o si es facil de leer.
## condicon 1. palabras raras.
## condicion 2. longitud de las oraciones
## condicion 3. sintaxis complicada.
## sintaxis complicada => subordinacion, varios verbos, elemento inicial por determinarse.

oracion_simple = "Amo al perro de mi abuela."
oracion_compleja = "llegamos que cocina detras de los arbustos que va a la escuela a cocinar para los adultos"

def lexical_complex_weighter(tagged_corpora):
    file = open("palabras.txt")
    data = file.readlines()
    file.close()
    weight = 0
    spec_words = [word.strip("\n") for word in data]
    for token in tagged_corpora:
        if token in spec_words:
            weight += 0
    return weight


def word_count_weighter(tagged_corpora):
    weight = len(tagged_corpora)
    return weight


def syntax_weighter(tagged_corpora):
    weight = 0
    for word in tagged_corpora:
        if re.search('V.+', word[1]):
            weight += 1
        if word[1] == 'PR':
            weight += 1
    if not re.search('D.+', tagged_corpora[0][1]) or re.search('N.+', tagged_corpora[0][1]):
        weight += 1
    return weight


def tagger(raw_corpora):
    LAW_TAGGER_PKL = open("law_tagger.pkl", 'rb')
    law_tagger = load(LAW_TAGGER_PKL)
    LAW_TAGGER_PKL.close()
    tokens = [token.lower() for token in word_tokenize(raw_corpora)]
    tagged_corpora = law_tagger.tag(tokens)
    print(tagged_corpora)
    return tagged_corpora


def read_complexity(word_count_weight, syntax_weight, lexical_weight):
    ##complexity is defined by a ratio between word count, syntax and lexical
    complexity = (0.02 * (word_count_weight) + 0.6 * (syntax_weight) + 0.2 * (lexical_weight))

    if complexity <= 5:
        print("Comprensión alta")
    if 6 < complexity <= 10:
        print("Comprensión media")
    if 11 < complexity:
        print("Comprensión baja")
    return complexity

def paragraph_analyzer(paragraph):
    complexity = 0
    polarity = 0
    paragraph = parer_split(paragraph)
    for sentence in paragraph:
        # polarity = sentiment_analyzer_model(sentence) + polarity
        tagged_corpora = tagger(sentence)
        lexical = lexical_complex_weighter(tagged_corpora)
        word_count = word_count_weighter(tagged_corpora)
        syntax = syntax_weighter(tagged_corpora)
        # read_complexity(lexical, word_count, syntax)
        print(sentence)

    paragraph_complexity = complexity/len(paragraph)
    print("La complejidad de lectura de este parrafo es de: " + paragraph_complexity)
    if polarity < 0:
        dictamen = "negativo"
    else:
        dictamen = "positivo"
    print("La polaridad del parrafo es: " + dictamen)
    return paragraph_complexity
def parer_split(text):

        text = re.sub("\t", "", text)
        text = re.sub("\n", " ", text)
        text = re.sub("\s+", " ", text)
        text = re.sub("\.\S", " ", text)
        text = re.sub("\s$", "", text)
        text = re.sub("Lic\.", "licenciado", text)
        text = re.sub("lic\.", "licenciado", text)

        parsed = text.split(". ")
        return parsed
def main():

    #sentence taken from Tribunal de apelacion civil Costa Rica.

    paragraph = """TRIBUNAL DE APELACION CIVIL Y DE TRABAJO ZONA SUR (SEDE PEREZ ZELEDON) (Materia Civil).- A las  diecisiete horas y cuarenta y tres minutos del doce de marzo del aNo dos mil diecinueve.-
	Recurso de Apelacion interpuesto por la actora  Inversiones Madereras CaNo Seco S.A., contra la resolucion de las digito:digito horas del digito de mayo del digito, dictada por el Juzgado Civil y de Trabajo del II Circuito Judicial de la Zona Sur, dentro del proceso Ordinario establecido por Inversiones Madereras CaNo Seco S.A., quien es representado por el Lic. Ricardo Otarola Otarola contra Misty Coral S.A., representada por su abogado director el Lic. Luis Octavio Perez Baires. Se integra el Tribunal con los Jueces Alexander Somarribas Tijerino, Allan Montero Valerio y Karla Gabriela Montiel Chan, correspondiendo a esta ultima la redaccion del fallo de Segunda Instancia.
CONSIDERANDO:I. El seNor Juez del Juzgado Civil y de Trabajo del II Circuito Judicial de la Zona Sur, mediante resolucion de las digito: digito horas del digito de mayo del digito; resolvio: (...) Habiendose demostrado fehacientemente que se a admitido causa penal por el (los) delito(s) de Uso de Documento Falso cuya decision podria influir eventualmente en el presente asunto, de conformidad con el articulo digito Codigo Procesal Civil, se ordena la inmediata suspension de los procedimientos la cual no podra durar mas de dos aNos al cabo de los cuales se reanudara el proceso. En cuanto los escritos presentados por la parte actora en el legajo de prueba de la misma, ingresados al sistema con fechas de digito y digito de abril del digito, se proceden a reservar los mismos para ser resueltos una vez haya transcurrido el plazo determinado anteriormente.- (Sic)."""

    paragraph_analyzer(paragraph)
    pass


if __name__ == '__main__':
    main()
