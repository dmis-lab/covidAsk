"""
.. module:: BEST
    :platform: Unix, linux, Windows
.. moduleauthor:: Sunkyu Kim <sunkyu-kim@korea.ac.kr>

================================
Biomedical Entity Query API v2
================================

API Description
================
This API is for use of BEST(Biomedical Entity Search Tool) in various purposes.


All users can access BEST at : http://best.korea.ac.kr/

For bugs and inquiries, please contact:

 * Jaewoo Kang(kangj@korea.ac.kr)
 * Sunkyu Kim(sunkyu-kim@korea.ac.kr)

Reference : https://doi.org/10.1371/journal.pone.0164680


Usage Examples
===============
To see ‘gene’s related ‘breast cancer’, use this sample code.

>>> bestQuery = best.BESTQuery("breast cancer",
                                filterObjectName="gene",
                                noAbsTxt=False)
>>> searchResult = best.getRelevantBioEntities(bestQuery)
>>> print(searchResult)
[{  'entityname' : 'ERBB2',
    'score' : 8098.43,
    'PMIDs' : ['28427196', '28341751', '28199325'],
    'abstracts' : [
                'Molecular-based cancer tests...',
                'The molecular subtype of breast...'
                'Breast cancer is the second leading cause of...'],
    'numArticles':14537
    'rank' : 1},
 {  'entityname' : 'ESR1',
    'score' : 7340.54,
    'PMIDs' : ['27923387', '28274211', '26276891'],
    'abstracts' : [
                'Several studies have shown that mammographic..',
                'A shift towards less burdening and more...'
                'The complete molecular basis of the organ-...'],
    'numArticles':18084
    'rank' : 2},
    ...
]

Changing noAbsTxt=True can make the process faster.

>>> bestQuery = best.BESTQuery("breast cancer",
                                filterObjectName="gene",
                                noAbsTxt=True)
>>> searchResult = best.getRelevantBioEntities(bestQuery)
>>> print(searchResult)
[{  'entityname' : 'ERBB2',
    'score' : 8098.43,
    'PMIDs' : [],
    'abstracts' : [],
    'numArticles':14537
    'rank' : 1},
 {  'entityname' : 'ESR1',
    'score' : 7340.54,
    'PMIDs' : [],
    'abstracts' : [],
    'numArticles':18084
    'rank' : 2},
    ...
]

If you want to see other entity types, change filterObjectName.

.. note:: Total 10 filterObjects(entity types) are available.

 * gene
 * drug
 * chemical compound
 * target
 * disease
 * toxin
 * transcription factor
 * mirna
 * pathway
 * mutation

>>> bestQuery = best.BESTQuery("breast cancer",
                                filterObjectName="drug",
                                noAbsTxt=True)
>>> searchResult = best.getRelevantBioEntities(bestQuery)
>>> print(searchResult)
[{  'entityname' : 'tamoxifen',
    'score' : 3208.687,
    'abstracts' : [],
    'numArticles':10583
    'rank' : 1},
 {  'entityname' : 'doxorubicin',
    'score' : 1639.867,
    'abstracts' : [],
    'numArticles':6074
    'rank' : 2},
    ...
]

Class/Function Description
===========================
"""
import http
#from http.client import HTTPException
import socket

class BESTQuery():
    """
    BESTQuery class is basic query object for BEST API.

    """

    __besturl = "http://best.korea.ac.kr/s?"


    def __init__(self, querystr, filterObjectName="All Entity Type", topN=20, noAbsTxt=True):
        """BESTQuery
        :param querystr, filterObjectName : result type, topN, noAbsTxt : if True, the result doesn't include the abstract texts.
.
        >>> query = BESTQuery("lung cancer", filterObjectName="gene", topN=10, noAbsTxt=False)
        >>> # 10 genes related with lung cancer is searched including the abstract texts.
        """

        self.querystr = querystr
        self.filterObjectName = filterObjectName
        self.topN = topN
        self.noAbsTxt = noAbsTxt

    def setQuerystr (self, querystr):
        """Setting the query

        :param querystr: a string object

        >>> query.setQuery(["cancer"])
        """
        if type(querystr) is not str:
            print ("Initialize error : invalid query. It should be a string object.")
            print (querystr)
            return

        if len(querystr) == 0:
            return

        self.querystr = querystr

    def getQuerystr (self):
        """Getting the query String

        :return: A string

        >>> querystr = query.getQuerystr()
        >>> print (querystr)
        ["cancer"]
        """
        return self.querystr

    def _isValid(self):
        if self.querystr is not None and self.querystr is not None and type(self.querystr) is not str:
            return False

        for keya in self.querystr :
            if type(keya) is not str :
                return False

        if self.topN <= 0:
            return False

        return True

    def setTopN (self, n):
        """ Setting the number of results retrieved by query

        :param n: the number of results to be retrieved

        >>> query.setTopN(100)
        """
        self.topN = n

    def getTopN (self):
        """ Getting the number of results retrieved by query

        :return: the number of results to be retrieved

        >>> print (query.getTopN())
        100
        """
        return self.topN

    def setFilterObjectName (self, oname):
        """ Setting the filtering object.
        Total 10 types are available.

         * gene
         * drug
         * chemical compound
         * target
         * disease
         * toxin
         * transcription factor
         * mirna
         * pathway
         * mutation

        >>> qeury.setFilterObjectName("Gene")
        """
        self.filterObjectName = oname

    def getFilterObjectName (self):
        """ Getting the filtering entity type.

        >>> print(query.getFilterObjectName())
        "breast cancer"
        """
        return self.filterObjectName

    def makeQueryString(self):
        queryKeywords = self.querystr
        querytype = self.filterObjectName.lower()
        noAbsTxt = self.noAbsTxt

        import urllib.parse

        queryKeywords = "q=" + urllib.parse.quote(queryKeywords)

        otype = ""
        if querytype == "gene":
            otype = "8"
        elif querytype == "drug":
            otype = "5"
        elif querytype == "chemical compound":
            otype = "3"
        elif querytype == "target":
            otype = "14"
        elif querytype == "disease":
            otype = "4"
        elif querytype == "toxin":
            otype = "15"
        elif querytype == "transcription factor":
            otype = "16"
        elif querytype == "mirna":
            otype = "10"
        elif querytype == "pathway":
            otype = "12"
        elif querytype == "mutation":
            otype = "17"
        elif querytype == "all entity type":
            otype = ""
        else:
            print ("Invalid type! Object type : All Entity Type")
            otype = ""

        if noAbsTxt:
            strQuery = self.__besturl + "t=l&wt=xslt&tr=tmpl2.xsl" + "&otype=" + otype + "&rows=" + str(self.topN) + "&" + queryKeywords
        else:
            strQuery = self.__besturl + "t=l&wt=xslt&tr=tmpl_170602.xsl" + "&otype=" + otype + "&rows=" + str(self.topN) + "&" + queryKeywords

        return strQuery

    def toDataObj(self):
        return {"query":self.querystr, "filterObjectName":self.filterObjectName, "topN":self.topN}

def getRelevantBioEntities(bestQuery):
    """ Function for retrieval from BEST

    :param bestQuery: BESTQuery

    :return: parsed objects (dict-BIOENTITY).

    * BIOENTITY (dict): {"entityName":str, "rank":int, "score":float, "numArticles":int, "abstracts":[str]}

    >>> bestQuery = BESTQuery(  "lung cancer",
                                filterObjectName="gene",
                                topN=10,
                                noAbsTxt=True   )
    >>> relevantEntities = getRelevantBioEntities(bestQuery)

    """
    if not (type(bestQuery) is BESTQuery):
        print ("query is invalid! please check your query object.")
        return None

    if not bestQuery._isValid() :
        print ("Query object is invalid. Please check the query")
        print ("Query : ")
        print ("   query: " + str(bestQuery.query))
        print ("   topN: " + str(bestQuery.topN))

        return None

    urlquery = bestQuery.makeQueryString()

    import urllib.request

    resultStr = ""
    again = 0
    while(again < 5) :
        try:
            request = urllib.request.Request(urlquery)
            request.add_header('User-Agent', 'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)')

            geneUrl = urllib.request.urlopen(request, timeout=5)
            resultStr = geneUrl.read().decode('utf-8')
            again = 10
        except http.client.BadStatusLine:
            again += 1
        except http.client.HTTPException:
            again += 1
        except socket.timeout:
            again += 1
        except socket.error:
            again += 1
        except urllib.error.URLError:
            again += 1
        except Exception:
            again += 1

    if again == 5:
        print("Network status is not good")
        return None

    result = __makeDataFromBestQueryResult(resultStr)

    return result

def __makeDataFromBestQueryResult(resultStr):
    lines = resultStr.split('\n')
    linesCnt = len(lines)

    resultDataArr = []
    curData = {"rank":0}
    for i in range(1, linesCnt) :
        line = lines[i]

        if line.startswith("@@@"):
            pmid, text = line[3:].strip().split("###")
            curData["abstracts"].append(text)
            curData["PMIDs"].append(pmid)
        else:
            if len(line.strip()) == 0 :
                continue

            if curData["rank"] != 0:
                resultDataArr.append(curData)

            dataResult = line.split(" | ")

            curData = {"rank":int(dataResult[0].strip()), "entityName":dataResult[1].strip(), "score":float(dataResult[2].strip()), "numArticles":int(dataResult[3].strip()), "abstracts":[], "PMIDs":[]}

    resultDataArr.append(curData)

    return resultDataArr

