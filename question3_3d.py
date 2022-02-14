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
    div_QS = np.divide(Q1,S1)
    #div_QS = np.nan_to_num(div_QS,nan=0,posinf=0,neginf=0)
    log_QS = np.log(div_QS)
    #log_QS=np.nan_to_num(div_QS,nan=0,posinf=0,neginf=0)
    mul_QS=np.multiply(Q,log_QS)
    res = np.sum(mul_QS)
    return res    

def grider(img,grid):
    piece = grid*grid
    x=int(img.shape[0]/grid)
    y=int(img.shape[1]/grid)
    grid_image = np.zeros((piece,x,y,3),dtype=int)
    k=0
    for i in range(0,img.shape[0],x):
        for j in range(0,img.shape[1],y):
            a_grid=img[i:i+x,j:j+y]
            grid_image[k] = a_grid
            k=k+1
    
    return grid_image

if __name__=='__main__':
    dataset1,images_names1 = Dataloader("query_1")
    dataset2,images_names2 = Dataloader("query_2")
    dataset3,images_names3 = Dataloader("query_3")
    dataset_support,images_names_s = Dataloader("support_96")
    
    
    SIZE=len(dataset_support)
    result = np.zeros(SIZE)
    pred = np.zeros(SIZE)
    index = list(range(0,SIZE))
    
    
    interval = 16 #query1,2 change
    #interval = 64 #query3 change
    
    #grid_type=[12,16,24,48]
    grid_type=[8,6,4,2]
    
     
    for i in grid_type:
        
        hist_of_query=[]
        hist_of_support_patches=[]
        hist_of_support=[]
        
        for data in dataset1:     #change dataset
            hist_of_query_patches=[]
            patches = grider(data,i)
            for patch in patches:
                
                channels =ThreeD_Color_Histogram(patch,interval)
                hist_of_query_patches.append(channels)
            hist_of_query.append(hist_of_query_patches)
            
        for data in dataset_support:     
            hist_of_support_patches=[]
            patches = grider(data,i)
            for patch in patches:
                channels=ThreeD_Color_Histogram(patch,interval)
                hist_of_support_patches.append(channels)
            hist_of_support.append(hist_of_support_patches)
        j=0
        accuracy=0
        for Qhist1 in hist_of_query:
            w=0
            result = np.zeros(SIZE)
            for Shist1 in hist_of_support:
                for k in range(len(hist_of_query_patches)):
                    temp_result = KL_divergence(Qhist1[k],Shist1[k])
                    result[w]=result[w]+temp_result
                result[w] = result[w]/len(hist_of_query_patches)    
                w=w+1
            min = np.argmin(result)
            
            pred[j] = min
            if pred[j]==index[j]:
                accuracy=accuracy+1
            j=j+1
        acc=100*accuracy/SIZE
        
        with open('results.txt', 'a') as f:
            f.write("\nFor grid: {}\n".format(96//i))
            f.write("For interval: {}\n".format(interval))
            f.write("Accuracy: {}\n".format(acc))
        
        
        print("\nFor grid: {}\n".format(96//i))
        print("For interval: {}\n".format(interval))
        print("Accuracy: {}\n".format(acc))