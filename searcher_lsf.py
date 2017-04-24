import pickle
import linecache
import config


class Searcher:

    def __init__(self, model):
        # store our index path
        self.model = model
        self.lsf = {}
        self.nchunks = config.LSFModelInfo['nLSFModelTrees']
        for nt in xrange(0, self.nchunks):
            print("loading nt" + str(nt))
            self.lsf[nt] = (pickle.load(
                open(self.model + "/LSFModelsave.p" + str(nt), "rb")))
      # self.kdtree = pickle.load( open( model+"/LSFModelsavekd.p", "rb"
      # ) );

    def search(self, queryFeatures, limit=10):
        resultsf = {}
        for nt in xrange(0, self.nchunks):

            dist, results = self.lsf[nt].kneighbors(
                queryFeatures, n_neighbors=10)
        #    dist,results = self.kdtree.query(queryFeatures, k=10);
            results = results[0]
            for i in xrange(len(results)):
                fileName = linecache.getline(
                    self.model + config.LSFModelInfo['indexFile'] + str(nt), results[i] + 1)
                fileName = fileName.replace('[', '')
                fileName = fileName.replace(']', '')
                fileName = fileName.replace('\'', '')
                if(fileName == ''):
                    print results[i]
                    continue
                resultsf[fileName] = dist[0][i]
               # open the index file for reading
                # return our (limited) results

        resultsf = sorted([(v, k) for (k, v) in resultsf.items()])
        return resultsf[:limit]
