import spacy

nlp = spacy.load('en_core_web_lg')

#sentences=["The big black cat stared at the small dog.", "Jane watched her brother in the evenings."]
sentences=["Who is the APS director?", "Who is Jonathan Lang?", "What is AEcroscopy and CNMS?", "How can I perform a spiral scan using AEcroscopy?", "How can I perform spectroscopy using AEcroscopy? What are the available options?"]

def get_subject_phrase(doc):
    for token in doc:
        if ("subj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]

def get_object_phrase(doc):
    for token in doc:
        if ("dobj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]

def extract_proper_nouns(doc):
    pos = [tok.i for tok in doc if tok.pos_ == "PROPN"]
    consecutives = []
    current = []
    for elt in pos:
        if len(current) == 0:
            current.append(elt)
        else:
            if current[-1] == elt - 1:
                current.append(elt)
            else:
                consecutives.append(current)
                current = [elt]
    if len(current) != 0:
        consecutives.append(current)
    return [doc[consecutive[0]:consecutive[-1]+1] for consecutive in consecutives]


"""
for sentence in sentences:
    doc = nlp(sentence)
    subject_phrase = get_subject_phrase(doc)
    object_phrase = get_object_phrase(doc)
    print("Subject:", subject_phrase)
    print("Object:", object_phrase)

for sentence in sentences:
    doc = nlp(sentence)
    print(doc)
    for word in doc.ents:
        print(word.text, word.label_)
for sentence in sentences:
    doc = nlp(sentence)
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
"""
for sentence in sentences:
    doc = nlp(sentence)
    nouns = extract_proper_nouns(doc)
    subject_phrase = get_subject_phrase(doc)
    object_phrase = get_object_phrase(doc)

    if nouns is not None:
        nouns = [noun.text.strip() for noun in nouns]
        for noun in nouns:
            print("Nouns", noun, len(noun))

    if subject_phrase is not None:
        subject_phrase = subject_phrase.text.strip()
        print("Subject:", subject_phrase, len(subject_phrase))
    else : subject_phrase = ""

    if object_phrase is not None:
        object_phrase = object_phrase.text.strip()
        print("Object:", object_phrase, len(object_phrase))
    else : object_phrase = ""

    uniques = list(set(nouns + [subject_phrase] + [object_phrase]))
    print ("Merged list: ", uniques)
    uniques = list(filter(lambda i: len(i) >= 5 , uniques))
    print ("Filtered list: ", uniques)


