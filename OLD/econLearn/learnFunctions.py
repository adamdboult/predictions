####################
# Global functions #
####################

def getMaxIndex(array):
    max_val = max(array)
    max_inx = array.index(max_val)
    return (max_inx)

def dictionaryToVector(dictionary):
    vector = []
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            output = dictionaryToVector(value)
        else:
            output = value
        if isinstance(output, list):
            for item in output:
                vector.append(item)
        else:
            vector.append(output)
    return (vector)

def addObjects(original, update):
    for good in original:
        for VorP in original[good]:
            adjustment = update[good][VorP]
            original[good][VorP] = original[good][VorP] + adjustment

def getTemplate(templateType, goods):
    template = {}
    if (templateType == "goods"):
        for good in goods:
            template[good] = 0
        
    elif (templateType == "action"):
        for good in goods:
            template[good] = {
                "v": 0,
                "p": 0
            }
    elif (templateType == "order"):
        for good in goods:
            template[good] = []

    elif (templateType == "state"):
        goodsTemplate = getTemplate("goods", goods)
        actionTemplate = getTemplate("action", goods)
        template = {
            "exists": 0,
            "assets": goodsTemplate,
            "actions": actionTemplate,
            "prices": goodsTemplate,
            "accept": goodsTemplate
        }
    return (template)
