import numpy as np

def loacdataset(filename):
    data_matrix = []
    data_labels = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line_arr = line.strip().split('\t')
            data_matrix.append([float(line_arr[0]), float(line_arr[1])])
            data_labels.append(line_arr[-1])
    return data_matrix, data_labels


def select_j_rand(i, m):
    j = i
    with(j == i):
        j = int(np.random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj





'''以下没有径向基函数版本'''

class optStructwithoutK:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2))) #first column is valid flag

def select_j(i, opt_s, e_i):
    max_k = -1
    max_delta_e = 0
    e_j = 0
    opt_s.e_cache[i] = [1, e_i]
    valid_ecache_list = np.nonzero(opt_s.e_cache[:, 0].A)[0]  # 存疑 不用.A 也可以
    if len(valid_ecache_list) > 0:
        for k in valid_ecache_list:
            if k == i: continue
            e_k = calc_ek(opt_s, k)
            delta_e = np.abs(e_i, e_k)
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                e_j = e_k
        return max_k, e_j
    else:
        j = select_j_rand(i, opt_s.m)
        e_j = calc_ek(opt_s, j)
        return j, e_j


def calc_ek(oS, k):
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    ek = fXk - float(oS.labelMat[k])
    return ek


def innerLwithoutK(i, oS):
    Ei = calc_ek(oS, i)
    if ((oS.label_mat[i]*Ei < -oS.toler) and (oS.alphas[i] < oS.C)) or ((oS.label_mat[i]*Ei > oS.toler) and (oS.alphas[i] > 0)):
        j,Ej = select_j(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.label_mat[i] != oS.label_mat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.label_mat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clip_alpha(oS.alphas[j],H,L)
        update_ek(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.label_mat[j]*oS.label_mat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        update_ek(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.label_mat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.label_mat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.label_mat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.label_mat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0


def smoPwithoutK(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStructwithoutK(np.mat(dataMatIn), np.mat(classLabels, dtype='float64').transpose(), C, toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerLwithoutK(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerLwithoutK(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas



'''以下有径向基版本'''
class optStructK:
    def __init__(self, data_mat_in, class_labels, C, toler, ktup):
        self.X = data_mat_in
        self.label_mat = class_labels
        self.C = C
        self.toler = toler
        self.m = np.shape(data_mat_in)[0]
        self.alphas = np.mat(np.zeros(shape=(self.m, 1)))
        self.b = 0
        self.e_cache = np.mat(np.zeros(shape=(self.m, 2)))
        self.K = np.mat(np.zeros(shape=(self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], ktup)


def kernel_trans(X, A, ktup):
    '''
        内核转换
    :param X:
    :param A:
    :param ktup:
    :return:
    '''
    m, n = X.shape
    K = np.mat(np.zeros((m, 1))) #m维列向量
    if ktup[0] == 'lin':
        K = X * A.T
    elif ktup[0] == 'rbf':
        for row_index in range(m):
            delta_row = X[row_index, :] - A # shape 1*n
            K[row_index] = delta_row * delta_row.T
        K = np.exp(K/ (-1 * ktup[1]**2))
    else: raise NameError('参数错误')
    return K


def calc_ekK(opt_s, k):
    fx_k = float(np.multiply(opt_s.alphas, opt_s.label_mat).T * opt_s.K[:, k] + opt_s.b)
    e_k = fx_k - float(opt_s.label_mat[k])
    return e_k


def selectK_j(i, opt_s, e_i):
    max_k = -1
    max_delta_e = 0
    e_j = 0
    opt_s.e_cache[i] = [1, e_i]
    valid_ecache_list = np.nonzero(opt_s.e_cache[:, 0].A)[0] #存疑 不用.A 也可以
    if len(valid_ecache_list) > 0:
        for k in valid_ecache_list:
            if k == i: continue
            e_k = calc_ekK(opt_s, k)
            delta_e = np.abs(e_i - e_k)
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                e_j = e_k
        return max_k, e_j
    else:
        j = select_j_rand(i, opt_s.m)
        e_j = calc_ekK(opt_s, j)
        return j, e_j


def update_ekK(opt_s, k):#after any alpha has changed update the new value in the cache
    e_k = calc_ekK(opt_s, k)
    opt_s.e_cache[k] = [1, e_k]

def update_ek(opt_s, k):#after any alpha has changed update the new value in the cache
    e_k = calc_ek(opt_s, k)
    opt_s.e_cache[k] = [1, e_k]


def innerL(i, oS):
    Ei = calc_ekK(oS, i)
    if ((oS.label_mat[i]*Ei < -oS.toler) and (oS.alphas[i] < oS.C)) or ((oS.label_mat[i]*Ei > oS.toler) and (oS.alphas[i] > 0)):
        j,Ej = selectK_j(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
        if (oS.label_mat[i] != oS.label_mat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i]
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.label_mat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clip_alpha(oS.alphas[j],H,L)
        update_ekK(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.label_mat[j]*oS.label_mat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        update_ekK(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.label_mat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, i] - oS.label_mat[j]*(oS.alphas[j]-alphaJold)*oS.K[i, j]
        b2 = oS.b - Ej- oS.label_mat[i]*(oS.alphas[i]-alphaIold)*oS.K[i, j] - oS.label_mat[j]*(oS.alphas[j]-alphaJold)*oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0


def smoPK(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):    #full Platt SMO
    oS = optStructK(np.mat(dataMatIn), np.mat(classLabels, dtype=float).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas


def calcWs(alphas,dataArr,classLabels):
    X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w


def test_rbf(k1=1.3):
    '''
    用于测试 径向基函数
    :param k1:
    :return:
    '''
    ktup = ('rbf', k1)
    data_arr, label_arr = loacdataset('testSetRBF.txt')
    b, alphas = smoPK(data_arr, label_arr, 200, 0.0001, 1000, ktup)
    # 复制一份数据 、再换成矩阵类型
    data_mat = np.mat(data_arr); label_mat = np.mat(label_arr, dtype=float).transpose() #换成列向量
    # 提取支持向量的索引
    sv_ind = np.nonzero(alphas.A > 0)[0]   #alphas shape是m*1
    svs = data_mat[sv_ind]
    sv_labels = label_mat[sv_ind]
    print('这里有 %d个支持向量' % np.shape(sv_labels)[0])
    m, n = data_mat.shape
    error_count = 0
    # 做出预测
    for i in range(m):
        kernel_eval = kernel_trans(svs, data_mat[i, :], ktup)  # m*1
        predict = kernel_eval.T * np.multiply(sv_labels, alphas[sv_ind]) + b  # (m*1) * (1*m)
        if np.sign(predict) != np.sign(label_mat[i]):
            # todo 测试传narray 类型和 array类型
            error_count += 1
    print('训练时错误率为 %f' % float(error_count / m))

    data_arr, label_arr = loacdataset('testSetRBF2.txt')
    error_count = 0
    data_mat = np.mat(data_arr);
    label_mat = np.mat(label_arr, dtype=float).transpose()  # 换成列向量
    m, n = data_mat.shape
    for i in range(m):
        kernel_eval = kernel_trans(svs, data_mat[i, :], ktup)  # m*1
        predict = kernel_eval.T * np.multiply(sv_labels, alphas[sv_ind]) + b  # (m*1) * (1*m)
        if np.sign(predict) != np.sign(label_mat[i]):
            # todo 测试传narray 类型和 array类型
            error_count += 1
    print('测试时错误率为 %f' % float(error_count / m))



if __name__ == '__main__':
    test_rbf()








