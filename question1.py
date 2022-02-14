

from PIL import Image
import numpy as np



def Dataloader(folder):
    dataset = np.zeros((200,96,96,3),dtype=int)
    image_names =[]
    with open('InstanceNames.txt') as f:
        lines = f.readlines()
        i=0
        for l in lines:
            image_names.append(l)
            path = folder + "/" + l.strip("\n")
            im = Image.open(path) 
            #im.show()
            image = np.array(im)
            dataset[i] = image
            i=i+1
    return dataset,image_names

def Per_Channel_Color_Histogram(img,interval):
    size = int(256/interval)
    r_hist = np.zeros(size,dtype=int)
    g_hist = np.zeros(size,dtype=int)
    b_hist = np.zeros(size,dtype=int)
    
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            #red
            r_value = img[row][col][0]
            r_location = int(r_value//interval)
            r_hist[r_location] +=1
            #green
            g_value = img[row][col][1]
            g_location = int(g_value//interval)
            g_hist[g_location] +=1
            #blue
            b_value = img[row][col][2]
            b_location = int(b_value//interval)
            b_hist[b_location] +=1
            
    return r_hist, g_hist, b_hist 
    
def ThreeD_Color_Histogram(img,interval):
    size = int(256/interval)
    hist = np.zeros((size,size,size),dtype=int)
    
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            #red 
            r_value = img[row][col][0]
            r_location = int(r_value//interval)
            #green
            g_value = img[row][col][1]
            g_location = int(g_value//interval)
            #blue
            b_value = img[row][col][2]
            b_location = int(b_value//interval)
            
            hist[r_location][g_location][b_location] +=1
             
    return hist

def l1_norm(hist):
    sum = np.sum(hist)
    hist=hist/sum
    return hist

epsilon=10**(-5)
def KL_divergence(Q_hist,S_hist):
    size=len(Q_hist)
    Q=l1_norm(Q_hist)
    S=l1_norm(S_hist)
    Q1=Q+epsilon
    S1=S+epsilon
    #Q = np.where(Q==0,10**(-20),Q)
    #S = np.where(S==0,10**(-20),S)
    div_QS = np.divide(Q1,S1)
    #div_QS = np.nan_to_num(div_QS,nan=0,posinf=0,neginf=0)
    log_QS=np.log(div_QS)
    #log_QS=np.nan_to_num(div_QS,nan=0,posinf=0,neginf=0)
    mul_QS=np.multiply(Q,log_QS)
    res = np.sum(mul_QS)
    return res    


if __name__=='__main__':
    dataset1,images_names1 = Dataloader("query_1")
    dataset2,images_names2 = Dataloader("query_2")
    dataset3,images_names3 = Dataloader("query_3")
    dataset_support,images_names_s = Dataloader("support_96")
    SIZE=len(dataset_support)
    result = np.zeros(SIZE)
    
    #histogram_type=['2D','3D']
    #grid_bool=False
    #grid_type=[12,16,24,48]
    interval_3D = [16,32,64,128]
    interval_2D = [4,8,16,32,64]
    
    pred = np.zeros(SIZE)
    index = list(range(0,SIZE))
    
    
    
    for interval in interval_3D:
        
        hist_of_query1=[]
        hist_of_support=[]
        
        for query1 in dataset3: #change
            chanells=ThreeD_Color_Histogram(query1,interval)  
            hist_of_query1.append(chanells)
            
        for support1 in dataset_support: 
            chanells_S=ThreeD_Color_Histogram(support1,interval) 
            hist_of_support.append(chanells_S)
       
        accuracy=0
        j=0
        for Q_data in dataset3: #change 
            i=0
            for S_data in dataset_support:
                
                result_r = KL_divergence(hist_of_query1[j],hist_of_support[i])
             
                result[i] = result_r
                i=i+1
            
            min=np.argmin(result)
            pred[j]=min
            if pred[j]==index[j]:
                accuracy=accuracy+1
            j=j+1
            
        acc=100*accuracy/SIZE
        
        with open('results.txt', 'a') as f:
    
            f.write("\nFor interval: {}\n".format(interval))
            f.write("Accuracy: {}\n".format(acc))
        
        print("For interval: {}".format(interval))
        print("Accuracy: {}\n".format(acc))
        