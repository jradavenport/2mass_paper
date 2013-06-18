import MySQLdb
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import astroML.filters
import os, sys
import cPickle
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

db     = MySQLdb.connect(host='tddb.astro.washington.edu', user='tddb', passwd='tddb', db='2MASS')
cursor = db.cursor()

class TwoMassLc(object):
    def __init__(self, tmid, w1, w2, w3, period=None):
        self.tmid   = tmid
        self.w1w2   = w1-w2 if (w1>0 and w2>0) else np.inf
        self.period = period

        self.fillLc()
        self.fillStats()
        
    def getFeatures(self, includeWise=False):
        return np.array((np.log10(self.period), self.JHmedian, self.HKmedian,
                         self.sampJ, self.sampJH, self.sampHK,
                         self.sskewJ, self.sskewJH, self.sskewHK,
                         self.w1w2))
    def getNames(self):
        return ["log period", "median J-H", "median H-K",
                "smoothed amp J", "smoothed amp J-H", "smoothed amp H-K",
                "smoothed skew J", "smoothed skew J-H", "smoothed skew H-K",
                "w1 - w2"]
                
    def fillLc(self):
        sql = 'select * from source where (objectId=%s) order by TDB' % (self.tmid)
        cursor.execute(sql)
        results = cursor.fetchall()
     
        # only look at detections of quality A-C
        # AND IN ALL 3 PASSBANDS!!!
        good = ['A', 'B', 'C']
        tdb    = []
        Jmags  = []
        Jdmags = []
        Hmags  = []
        Hdmags = []
        Kmags  = []
        Kdmags = []
        
        for result in results:
            # For here just look at epochs where all 3 are good
            if (result[8] in good) and (result[16] in good) and (result[24] in good):
                tdb.append( result[4] )

                Jmags.append( result[6] )
                Jdmags.append( result[7] )
     
                Hmags.append( result[14] )
                Hdmags.append( result[15] )
     
                Kmags.append( result[22] )
                Kdmags.append( result[23] )
                
        self.tdb    = np.array(tdb)
        self.Jmags  = np.array(Jmags)
        self.Jdmags = np.array(Jdmags)
        self.Hmags  = np.array(Hmags)
        self.Hdmags = np.array(Hdmags)
        self.Kmags  = np.array(Kmags)
        self.Kdmags = np.array(Kdmags)

    def binData(self, x, y, dy, dbin=0.01):
        bins = np.arange(0, 1.0+dbin, dbin)
        bx   = []
        by   = []
        bdy  = []
        for i in range(1, len(bins)):
            idx = np.where( (x > bins[i-1]) & (x < bins[i]) )
            bx.append(0.5 * (bins[i] + bins[i-1]))

            # Weighted Mean
            numer = np.sum(y[idx] / dy[idx]**2)
            denom = np.sum(1. / dy[idx]**2)
            wmean = numer / denom
            wrms  = np.sqrt(1. / denom)

            by.append(wmean)
            bdy.append(wrms)
        return np.array(bx), np.array(by), np.array(bdy)

    def fillStats(self):
        JH            = self.Jmags - self.Hmags
        HK            = self.Hmags - self.Kmags
        self.JHmedian = np.median(JH)
        self.HKmedian = np.median(HK)
        
        # From data
        self.ampJ     = np.percentile(self.Jmags, 95) - np.percentile(self.Jmags, 5)
        self.ampJH    = np.percentile(JH, 95) - np.percentile(JH, 5)
        self.ampHK    = np.percentile(HK, 95) - np.percentile(HK, 5)
        self.skewJ    = scipy.stats.skew(self.Jmags)
        self.skewJH   = scipy.stats.skew(JH)
        self.skewHK   = scipy.stats.skew(HK)

        # Folded and binned
        if self.period is not None:
            phase         = self.tdb / self.period - self.tdb // self.period
            idx           = np.argsort(phase)
            phase         = phase[idx]

            J             = self.Jmags[idx]
            dJ            = self.Jdmags[idx]

            JH            = JH[idx]
            dJH           = np.sqrt(self.Jdmags**2 + self.Hdmags**2)[idx]

            HK            = HK[idx]
            dHK           = np.sqrt(self.Hdmags**2 + self.Kdmags**2)[idx]

            sJx,  sJ,  sdJ  = self.binData(phase, J,  dJ)
            sJHx, sJH, sdJH = self.binData(phase, JH, dJH)
            sHKx, sHK, sdHK = self.binData(phase, HK, dHK)

            #sJ            = astroML.filters.savitzky_golay(np.ravel((sJ,sJ,sJ)),    window_size=2*int(0.005*len(sJ))+1,  order=3)[len(sJ):2*len(sJ)]
            #sJH           = astroML.filters.savitzky_golay(np.ravel((sJH,sJH,sJH)), window_size=2*int(0.005*len(sJH))+1, order=3)[len(sJH):2*len(sJH)]
            #sHK           = astroML.filters.savitzky_golay(np.ravel((sHK,sHK,sHK)), window_size=2*int(0.005*len(sHK))+1, order=3)[len(sHK):2*len(sHK)]

            self.sampJ    = np.max(sJ)  - np.min(sJ)
            self.sampJH   = np.max(sJH) - np.min(sJH)
            self.sampHK   = np.max(sHK) - np.min(sHK)
            self.sskewJ   = scipy.stats.skew(sJ)
            self.sskewJH  = scipy.stats.skew(sJH)
            self.sskewHK  = scipy.stats.skew(sHK)
    
            if False:
                plt.figure()
                plt.plot(phase, J, "ro", alpha=0.1)
                plt.plot(sJx, sJ)
    
                plt.figure()
                plt.plot(phase, JH, "ro", alpha=0.1)
                plt.plot(sJHx, sJH)
    
                plt.figure()
                plt.plot(phase, HK, "ro", alpha=0.1)
                plt.plot(sHKx, sHK)
    
        else:
            self.sampJ    = None
            self.sampJH   = None
            self.sampHK   = None
            self.sskewJ   = None
            self.sskewJH  = None
            self.sskewHK  = None
            self.dampJ    = None
            self.dampJH   = None
            self.dampHK   = None

def plotClasses(x, featureNames, classLabels, classNames, plot_colors = "bry"):
    figs = []
    for idx, pair in enumerate(itertools.combinations(np.arange(x.shape[1]), 2)):
        data  = x[:, pair]
        mean  = data.mean(axis=0)
        std   = data.std(axis=0)
        ndata = (data - mean) / std
        clf1  = DecisionTreeClassifier().fit(data,  classLabels)
        clf2  = RandomForestClassifier(n_estimators=10).fit(data,  classLabels)
        clf3  = DecisionTreeClassifier().fit(ndata, classLabels)
        clf4  = RandomForestClassifier(n_estimators=10).fit(ndata, classLabels)
        
        fig   = plt.figure()
        # Un-normalized
        sp1   = fig.add_subplot(221)
        sp2   = fig.add_subplot(222)
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z1 = clf1.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        sp1.contourf(xx, yy, Z1, cmap=plt.cm.Paired)

        for tree in clf2.estimators_:
            Z2 = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            sp2.contourf(xx, yy, Z2, alpha=0.1, cmap=plt.cm.Paired)
        sp1.set_ylabel(featureNames[pair[1]])
        sp1.set_title("DecisionTreeClassifier")
        sp2.set_title("RandomForestClassifier")
        sp1.axis("tight")
        sp2.axis("tight")

        for i, color in zip(xrange(len(classNames)), plot_colors):
            idx = np.where(classLabels == i)
            sp1.scatter(data[idx, 0], data[idx, 1], c=color, label=classNames[i],
                        cmap=plt.cm.Paired)
            sp2.scatter(data[idx, 0], data[idx, 1], c=color, label=classNames[i],
                        cmap=plt.cm.Paired)


        # Normalized
        sp3   = fig.add_subplot(223)
        sp4   = fig.add_subplot(224)
        x_min, x_max = ndata[:, 0].min() - 1, ndata[:, 0].max() + 1
        y_min, y_max = ndata[:, 1].min() - 1, ndata[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z3 = clf3.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        sp3.contourf(xx, yy, Z3, cmap=plt.cm.Paired)

        for tree in clf4.estimators_:
            Z4 = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            sp4.contourf(xx, yy, Z4, alpha=0.1, cmap=plt.cm.Paired)
        sp3.set_ylabel(featureNames[pair[1]])
        sp3.set_title("DecisionTreeClassifier")
        sp4.set_title("RandomForestClassifier")
        sp3.axis("tight")
        sp4.axis("tight")
        sp3.set_xlabel(featureNames[pair[0]])
        sp3.set_ylabel(featureNames[pair[1]])
        sp4.set_xlabel(featureNames[pair[0]])

        for i, color in zip(xrange(len(classNames)), plot_colors):
            idx = np.where(classLabels == i)
            sp3.scatter(ndata[idx, 0], ndata[idx, 1], c=color, label=classNames[i],
                        cmap=plt.cm.Paired)
            sp4.scatter(ndata[idx, 0], ndata[idx, 1], c=color, label=classNames[i],
                        cmap=plt.cm.Paired)
    
        print "DT SCORE:", featureNames[pair[0]], "vs", featureNames[pair[1]], ":",
        print cross_val_score(clf1, data, classLabels), 
        print cross_val_score(clf3, ndata, classLabels)
        print "RF SCORE:", featureNames[pair[0]], "vs", featureNames[pair[1]], ":",
        print cross_val_score(clf2, data, classLabels), 
        print cross_val_score(clf4, ndata, classLabels)

        figs.append(fig)

    # All data
    clf1 = DecisionTreeClassifier().fit(x, classLabels)
    mean = x.mean(axis=0)
    std  = x.std(axis=0)
    nx   = (x - mean) / std
    clf2 = DecisionTreeClassifier().fit(nx, classLabels)
    print "DT SCORE ALL:", cross_val_score(clf1, x, classLabels), 
    print cross_val_score(clf2, nx, classLabels)
    print "  Feature importance:", clf1.feature_importances_
    print "  Feature importance:", clf2.feature_importances_

    # All data
    clf1 = RandomForestClassifier(n_estimators=10).fit(x, classLabels)
    clf2 = RandomForestClassifier(n_estimators=10).fit(nx, classLabels)
    print "RF SCORE ALL:", cross_val_score(clf1, x, classLabels), 
    print cross_val_score(clf2, nx, classLabels)
    print "  Feature importance:", clf1.feature_importances_
    print "  Feature importance:", clf2.feature_importances_

    return figs


# Note that we don't have the new objects in these files yet
# Once per_objs.lis has classes we have to redo this.
filename = "stats2.pickle"
if not os.path.isfile(filename):
    binaries  = []
    for line in open("objid_bb.lis").readlines()[1:]:
        if line.startswith("#"):
            continue
        fields = line.split()
        tmid   = fields[0]
        period = float(fields[1])
        w1     = float(fields[5])
        w2     = float(fields[6])
        w3     = float(fields[7])
        lc     = TwoMassLc(tmid, w1, w2, w3, period)
        binaries.append(lc)
    
    pulsators = []
    for line in open("objid_rr.lis").readlines()[1:]:
        fields = line.split()
        tmid   = fields[0]
        period = float(fields[1])
        w1     = float(fields[5])
        w2     = float(fields[6])
        w3     = float(fields[7])
        lc     = TwoMassLc(tmid, w1, w2, w3, period)
        pulsators.append(lc)

    # Not enough information in objid_ll.lis
    
    import cPickle
    buff = open(filename, "wb")
    cPickle.dump((binaries, pulsators), buff)
    buff.close()

else:

    buff = open(filename, "rb")
    binaries, pulsators = cPickle.load(buff)
    buff.close()


trainingSample = []
classLabels    = []
for s in binaries:
    f = s.getFeatures()
    if not (False in np.isfinite(f)):
        trainingSample.append(f)
        classLabels.append(0)

for s in pulsators:
    f = s.getFeatures()
    if not (False in np.isfinite(f)):
        trainingSample.append(f)
        classLabels.append(1)
trainingSample = np.array(trainingSample)
classLabels    = np.array(classLabels)
featureNames   = pulsators[0].getNames()
classNames     = ["Periodic", "Pulsators"]
plot_colors    = "bry"

# Do the machine learning!  

# Feature selection (what is important)
trainingSample_svc = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(trainingSample, classLabels)
print "SVC", trainingSample.shape, "->", trainingSample_svc.shape

clf = ExtraTreesClassifier(random_state=0)
trainingSample_etc = clf.fit(trainingSample, classLabels).transform(trainingSample)
print "ETC", trainingSample.shape, "->", trainingSample_etc.shape, clf.feature_importances_

# To capture all the plots:
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('classification.pdf')


figs = plotClasses(trainingSample, featureNames, classLabels, classNames)
#plt.show()
for fig in figs:
    pp.savefig(fig)

figs = plotClasses(trainingSample_svc, ["SVC%d"%(x) for x in xrange(trainingSample_svc.shape[1])], classLabels, classNames)
#plt.show()
for fig in figs:
    pp.savefig(fig)

figs = plotClasses(trainingSample_etc, ["ETC%d"%(x) for x in xrange(trainingSample_etc.shape[1])], classLabels, classNames)
#plt.show()
for fig in figs:
    pp.savefig(fig)

pp.close()
import pdb; pdb.set_trace()
    

decisionTree   = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0).fit(
    trainingSample, classLabels)


randomForest   = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0).fit(
    trainingSample, classLabels)
extraTrees     = ExtraTreesClassifier(n_estimators=50, max_depth=None, min_samples_split=1, random_state=0).fit(
    trainingSample, classLabels)
gradientBoost  = GradientBoostingClassifier(n_estimators=50, max_depth=1, learn_rate=1.0, random_state=0).fit(
    trainingSample, classLabels)

for classifier, name in ((decisionTree, "Decision Tree"),
                         (randomForest, "Random Forest"),
                         (extraTrees, "Extra Trees"),
                         (gradientBoost, "Gradient Boosting")):
    
    print "SCORE", name, cross_val_score(classifier, trainingSample, classLabels)

sys.exit(1)















if False:    
    bp = np.array([np.log10(x.period) for x in binaries])
    ba = np.array([np.log10(x.sampJ)  for x in binaries])
    bc = np.array([x.JHmedian         for x in binaries])
    bs = np.array([x.skewJ            for x in binaries])
    
    pp = np.array([np.log10(x.period) for x in pulsators])
    pa = np.array([np.log10(x.sampJ)  for x in pulsators])
    pc = np.array([x.JHmedian         for x in pulsators])
    ps = np.array([x.skewJ            for x in pulsators])
    
    plt.figure()
    plt.plot(bp, ba, 'ro')
    plt.plot(pp, pa, 'bs')
    plt.xlabel("log(Period / days)")
    plt.ylabel("log(J-band amplitude / mags)")
    
    
    plt.figure()
    #plt.scatter(np.append(bc,pc), np.append(bp,pp), s=50+50*np.append(bs,ps), c=np.append(ba,pa))
    plt.scatter(bc, bp, s=50+50*bs, c=ba, marker="s", alpha=0.5)
    plt.scatter(pc, pp, s=50+50*ps, c=pa, marker="o", alpha=0.5)
    plt.xlabel("J-H")
    plt.ylabel("log(Period / days)")
    plt.colorbar()
    
    plt.figure()
    plt.scatter(bc, bs, s=100+75*bp, c=ba, marker="s", alpha=0.9, zorder=2)
    plt.scatter(pc, ps, s=100+75*pp, c=pa, marker="o", alpha=0.9, zorder=3)
    l1 = plt.scatter(0, 0, s=100+75*-1, c=-0.9, marker="s", alpha=0.9, zorder=2, label="log(P) = -1")
    l2 = plt.scatter(0, 0, s=100+75*0, c=-0.9, marker="s", alpha=0.9, zorder=2, label="log(P) = 0")
    l3 = plt.scatter(0, 0, s=100+75*1, c=-0.9, marker="s", alpha=0.9, zorder=2, label="log(P) = +1")
    plt.legend(fancybox=True, shadow=True, numpoints=1)
    
    plt.figure()
    sql = 'select J_median-H_median,J_skew from object where (J_skew is not NULL)'
    cursor.execute(sql)
    results = cursor.fetchall()
    plt.plot([x[0] for x in results], 
             [x[1] for x in results], c="0.75", marker=".", ms=2, ls='None', alpha=0.75, zorder=1, label="__nolegend__")
    plt.scatter(bc, bs, s=100+75*bp, c=ba, marker="s", alpha=0.9, zorder=2, label="__nolegend__")
    plt.scatter(pc, ps, s=100+75*pp, c=pa, marker="o", alpha=0.9, zorder=3, label="__nolegend__")
    plt.xlabel("J-H")
    plt.ylabel("Skew (mag)")
    c = plt.colorbar()
    c.set_label("Amplitude (mag)", rotation=90)
    
    # set up 3 points to be covered by legend
    plt.legend((l1, l2, l3), (l1.get_label(), l2.get_label(), l3.get_label()),
               loc=7, fancybox=True, shadow=True, scatterpoints=1)
    
    plt.xlim(0.0, 1.0)
    plt.ylim(-1.0, 3.5)
    plt.show()
    
