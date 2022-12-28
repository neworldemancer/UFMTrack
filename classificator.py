from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

def _load_pckl(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data

def _save_pckl(d, fname):
    with open(fname, 'wb') as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
        
class Classificator():
    """
    Wrapper for log. regresion from raw data, keeping track of proper (training data) normalization,
    all params stored/loaded from a File for production.
    Also fits (`train`) the regressor.
    """
    def __init__(self):
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None
        
        self._clf = LogisticRegression()
        
        self._norm = (None, None)
        self._val_accuracy = None
        self._tra_accuracy = None
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return (f'Classificator based on {self._clf.__class__},\n\t'+
                f' t/v {len(self._y_train)}/{len(self._y_test)} samples.\n\t'+
                f' Accuracy {self._tra_accuracy:.3f} / {self._val_accuracy:.3f}')
    
    def _set_norm(self, data_samples):
        norm_m = data_samples.mean(axis=0, keepdims=True)
        norm_s = data_samples.std(axis=0, keepdims=True)
        data_samples_n = (data_samples-norm_m)/norm_s
        
        self._norm = (norm_m, norm_s)
        return data_samples_n
    
    def _norm_data(self, samples):
        norm_m, norm_s = self._norm
        
        return (samples-norm_m) / norm_s
    
    def predict(self, samples):
        samples_n = self._norm_data(samples)
        pred = self._clf.predict(samples_n)
        return pred
    
    def predict_proba(self, samples):
        samples_n = self._norm_data(samples)
        pred_proba = self._clf.predict_proba(samples_n)
        return pred_proba
    
    def save(classificator_instance, filename):
        _save_pckl(classificator_instance, filename)
        
    @staticmethod 
    def load(filename):
        classificator_instance:Classificator = _load_pckl(filename)
        return classificator_instance
        
    def train(self, X, y, test_size):
        """
        Normalizes X, trains logistic regressor on data (X_n, y), stores normalizations, and evaluates performance
        """
        X_n = self._set_norm(X)
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X_n, y, test_size=0.20, stratify=y)
        
        self._clf = self._clf.fit(self._X_train, self._y_train)
        
        self._tra_accuracy = self._clf.score(self._X_train, self._y_train)
        self._val_accuracy = self._clf.score(self._X_test, self._y_test)

def _test_():
    c = Classificator()

    mask_known = classes != c_unknown

    X = vectorized[mask_known]

    classes_known = classes[mask_known]
    is_detached = (classes_known==c_det)
    detached_class = is_detached.astype(np.int32)

    y = detached_class


    c.train(X, y, test_size=0.2)
    
    print(c)
    
    c.save('test_classifier.lrp')
    
    c1 = Classificator.load('test_classifier.lrp')
    
    print(c1)
    
    print ('saved vs loaded:', np.all(c1._clf.predict(c1._X_test) == c._clf.predict(c._X_test) ))