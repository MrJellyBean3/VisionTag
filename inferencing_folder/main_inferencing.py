import os, sys
import cv2
import numpy as np
import time
import keyboard
import multiprocessing

p1_ims_dir=r"D:\vscode\mmlab1\el1_ims"
p2_ims_dir=r"D:\vscode\mmlab1\el2_ims"


if __name__=="__main__":

    #move all images from the images folder to the images folder in the dataset folder
    

    headshots_on=False
    damage_headshot=100
    damage_mult_b=24

    damage_headshot_p1=90#m4a4    
    damage_mult_b_p1=30
    firerate_p1=11
    mag_size_p1=20

    damage_headshot_p2=100#ak
    damage_mult_b_p2=33
    firerate_p2=10
    mag_size_p2=20

def new_in_lists(que,new_l1,old_l1,new_l2,old_l2):
    for item in new_l1:
        if item not in old_l1:
            que.append(p1_ims_dir+"\\"+item)
    for item in new_l2:
        if item not in old_l2:
            que.append(p2_ims_dir+"\\"+item)
    return(que)


def file_que_handler(shared_que_info):
    print("starting file que thread")
    global p1_ims_dir
    global p2_ims_dir
    p1_ims_list=os.listdir(p1_ims_dir)
    p2_ims_list=os.listdir(p2_ims_dir)
    p1_ims_list_prev=p1_ims_list
    p2_ims_list_prev=p2_ims_list
    t_print=time.time()
    signal_prev=shared_que_info["signal"]
    local_que=[]
    while(True):
        if signal_prev!=shared_que_info["signal"]:
            local_que=[]
            signal_prev=shared_que_info["signal"]
        t_top_while=time.time()
        p1_ims_list=os.listdir(p1_ims_dir)
        p2_ims_list=os.listdir(p2_ims_dir)
        local_que=new_in_lists(local_que,p1_ims_list,p1_ims_list_prev,p2_ims_list,p2_ims_list_prev)
        shared_que_info["file_que_new"]=local_que
        p1_ims_list_prev=p1_ims_list
        p2_ims_list_prev=p2_ims_list
        
        
        
        if (time.time()-t_print)>5:
            print("file que loop time: ", time.time()-t_top_while)
            t_print=time.time()


def view_handler(shared_data):
    import pygame
    import pygame.mixer
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    import tkinter as tk
    riddle_list=[""]
    riddle_answers=[""]


    mask_path=r"D:\vscode\mmlab1\only_mask"+"\\"#mask dirs
    timestr = time.strftime("%m-%d-%Y-%H-%M-%S")
    mask_dir=mask_path+timestr
    os.mkdir(mask_dir)
    EL1_mask=mask_dir + "\\" + "EL1"
    EL2_mask=mask_dir + "\\" + "EL2"
    os.mkdir(EL1_mask)
    os.mkdir(EL2_mask)

    with open ("riddles.txt","r") as riddle_file:
        i=0
        for line in riddle_file:
            line=line.strip()
            if (i%2)==0:
                riddle_list.append(line)
            else:
                riddle_answers.append(line)
            i+=1
    correct_state=True
    correct_state_prev=True

    def submit_answer(answer,correct_answer):
        global correct_state
        #case insensitive
        if  answer.lower()==correct_answer.lower():
            result_label.configure(text="Correct")
            correct_state=True
        else:
            result_label.configure(text="Incorrect")

    #create tkinter window
    root = tk.Tk()
    root.title("Riddle Game")
    root.geometry("500x500")
    root.configure(bg='black')

    #display options for selecting a riddle
    riddle_selected=False
    riddle_var=tk.StringVar(root)
    riddle_var.set(riddle_list[0])
    riddle_menu=tk.OptionMenu(root,riddle_var,*riddle_list) 
    riddle_menu.config(height=5,width=100,font=("gameovercre", 18, "bold"))

    #wrap text
    riddle_menu.config(wraplength=400)
    riddle_menu.pack()


    #display entry box for answer
    answer_entry=tk.Entry(root)
    answer_entry.configure(width=20)
    answer_entry.config(font=("gameovercre", 24, "bold"))
    answer_entry.pack()
    #display button for submitting answer
    submit_button=tk.Button(root,text="Submit",command=lambda: submit_answer(answer_entry.get(),riddle_answers[riddle_list.index(riddle_var.get())]))
    #make it bigger
    submit_button.config(height=5,width=20)
    submit_button.pack()
    #display label for displaying result
    result_label=tk.Label(root,text="Result")
    result_label.config(height=5,width=20)
    result_label.pack()
    prev_riddle=riddle_var.get()

    #initialize pygame mixer
    pygame.mixer.init()
    #set pygame channel to 0
    pygame.mixer.Channel(0).set_volume(1.5)
    bomb_mode=True
    read_prev_value=0

    plt.ion()#setup plotting
    temp_img=[[[3,3,3]]]
    plt_img=plt.imshow(temp_img)
    temp_img=np.random.rand(480,640,3)
    plt_img.set_data(temp_img)
    p1_shot_count,p1_hit_count,p1_death_count=shared_data["stats_p1"]#shots hits deaths
    p2_shot_count,p2_hit_count,p2_death_count=shared_data["stats_p2"]

    p1_hit_count_prev=p1_hit_count
    p2_hit_count_prev=p2_hit_count
    
    t_u=time.time()
    p2_dead=False
    masks_saved_count=0
    current_result_save_dir=r"D:\vscode\mmlab1\results_saved"+"\\"+str(timestr)
    os.mkdir(current_result_save_dir)
    os.mkdir(current_result_save_dir+r"\hits")
    os.mkdir(current_result_save_dir+r"\misses")
    os.mkdir(current_result_save_dir+r"\hits\p1")
    os.mkdir(current_result_save_dir+r"\hits\p2")
    os.mkdir(current_result_save_dir+r"\misses\p1")
    os.mkdir(current_result_save_dir+r"\misses\p2")
    while True:
        if bomb_mode:#bomb display:
            if (time.time()-t_u)>0.25:
                if not p2_dead:
                    root.update()
                t_u=time.time()
            if riddle_var.get()!=prev_riddle:
                riddle_menu.configure(state="disabled")
                correct_state=False
            prev_riddle=riddle_var.get()
            
            if correct_state_prev and not correct_state:
                start_bomb=time.time()
                pygame.mixer.Channel(0).play(pygame.mixer.Sound("bomb2.mp3"))

            #if correct state or if music is done playing
            music_finished=pygame.mixer.music.get_busy()==0
            if correct_state:
                #turn off channel 0 sound
                pygame.mixer.Channel(0).stop()
                correct_state=True
            
            if not correct_state_prev and correct_state:
                riddle_menu.configure(state="active")
            correct_state_prev=correct_state


        if shared_data["read"]!=read_prev_value:#check for new image
            read_prev_value
            shared_data["read"]=0
            read_prev_value=shared_data["read"]
            dot_size=5
            try:
                im_data=cv2.imread(shared_data["img"])     #plotting
                img=shared_data["img"]
                if(len(img)>0):#determine who is shooting
                    if(img[19]=='1'):
                        p1_shooting=True
                    else:
                        p1_shooting=False
                im_data=cv2.cvtColor(im_data,cv2.COLOR_BGR2RGB)
                try:#get data from main
                    #result=shared_data["result"]
                    im_mask=shared_data["mask"]
                    shot_acc=shared_data["acc"].item()
                    got_mask=True
                except:
                    im_mask=[]
                    #result=[[[[0,0,0,0,0]]]]
                    shot_acc=0.00
                    got_mask=False

                if p1_shooting: #determine if hit or miss for results saving
                    if shot_acc>0.3:
                        results_dir=r"D:\vscode\mmlab1\results_saved"+"\\"+str(timestr)+r"\hits\p1"
                    else:
                        results_dir=r"D:\vscode\mmlab1\results_saved"+"\\"+str(timestr)+r"\misses\p1"
                else:
                    if shot_acc>0.3:
                        results_dir=r"D:\vscode\mmlab1\results_saved"+"\\"+str(timestr)+r"\hits\p2"
                    else:
                        results_dir=r"D:\vscode\mmlab1\results_saved"+"\\"+str(timestr)+r"\misses\p2"
                p1_hit_count_prev=shared_data["p1_hit_count"]
                p2_hit_count_prev=shared_data["p2_hit_count"]

                
                
                file_inc=0
                if shared_data["headshot"]:
                    while (os.path.isfile(results_dir+"\\"+str(img.split("\\")[-1])[0:-4]+"_acc"+str(int(100*shot_acc))+"_hs_"+str(file_inc)+".JPG")):#get save path for file
                        file_inc+=1
                    file_name=str(results_dir+"\\"+str(img.split("\\")[-1])[0:-4]+"_acc"+str(int(100*shot_acc))+"_hs_"+str(file_inc)+".JPG")
                else:
                    while (os.path.isfile(results_dir+"\\"+str(img.split("\\")[-1])[0:-4]+"_acc"+str(int(100*shot_acc))+"_"+str(file_inc)+".JPG")):#get save path for file
                        file_inc+=1
                    file_name=str(results_dir+"\\"+str(img.split("\\")[-1])[0:-4]+"_acc"+str(int(100*shot_acc))+"_"+str(file_inc)+".JPG")

                

                try:
                    
                    mask3=im_mask.cpu().numpy()
                    dims_t=mask3.shape
                    dims=(dims_t[0],dims_t[1])
                    print(dims,im_data.shape)
                    out_data=cv2.resize(im_data,(dims[1],dims[0]))
                    mask3=mask3[:,:,np.newaxis]
                    #mask3=mask3[:,:,np.newaxis]
                    #mask3=mask3[:,:,np.newaxis]

                    #make masked image where the masked area is the regular image and the rest is black
                    mask_image=np.zeros((dims[0],dims[1],3))
                    mask_image[:,:,2]=mask3[:,:,0]*out_data[:,:,0]
                    mask_image[:,:,1]=mask3[:,:,0]*out_data[:,:,1]
                    mask_image[:,:,0]=mask3[:,:,0]*out_data[:,:,2]

                    masks_saved_count+=1
                    if p1_shooting:
                        if shared_data["headshot"]:
                            cv2.imwrite(EL1_mask+"\\"+str(masks_saved_count)+"_"+str(int(shot_acc*100))+"_hs.jpg",mask_image)
                        else:
                            cv2.imwrite(EL1_mask+"\\"+str(masks_saved_count)+"_"+str(int(shot_acc*100))+".jpg",mask_image)
                    else:
                        if shared_data["headshot"]:
                            cv2.imwrite(EL2_mask+"\\"+str(masks_saved_count)+"_"+str(int(shot_acc*100))+"_hs.jpg",mask_image)
                        else:
                            cv2.imwrite(EL2_mask+"\\"+str(masks_saved_count)+"_"+str(int(shot_acc*100))+".jpg",mask_image)


                    out_data=out_data*(mask3+0.5)

                    for i in range(dot_size):
                        for j in range(dot_size):
                            out_data[int(dims[0]/4)+int(dot_size/2)+j][int(dims[1]/2)+int(dot_size/2)+i]=[255,255,255]
                            out_data[int(dims[0]*3/4)+int(dot_size/2)+j][int(dims[1]/2)+int(dot_size/2)+i]=[255,255,255]
                    plt_img.set_data(out_data/255)
                    #convert out_data to bgr and save it to saving_data without depth is 6 error
                    
                    cv2.imwrite(file_name,out_data)
                    out_data=cv2.imread(file_name)
                    saving_data=(cv2.cvtColor(out_data, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(file_name,saving_data)
                    #cv2.cvtColor(out_data, cv2.COLOR_RGB2BGR))
                    #cv2.imwrite(str(results_dir+r"\2.JPG"),cv2.cvtColor(out_data, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    t,o,b=sys.exc_info()
                    print(e, b.tb_lineno)
                    im_data=cv2.resize(im_data,(dims[1],dims[0]))
                    for i in range(dot_size):
                        for j in range(dot_size):
                            im_data[int(dims[0]/4)+int(dot_size/2)+j][int(dims[1]/2)+int(dot_size/2)+i]=[255,255,255]
                            im_data[int(dims[0]*3/4)+int(dot_size/2)+j][int(dims[1]/2)+int(dot_size/2)+i]=[255,255,255]
                    # im_data[int(dims[0]/4)][int(dims[1]/2)]=[0,0,0]
                    # im_data[int(dims[0]/4)+1][int(dims[1]/2)]=[510,510,510]
                    # im_data[int(dims[0]/4)-1][int(dims[1]/2)]=[510,510,510]
                    # im_data[int(dims[0]/4)][int(dims[1]/2)+1]=[510,510,510]
                    # im_data[int(dims[0]/4)][int(dims[1]/2)-1]=[510,510,510]
                    # im_data[int(dims[0]*3/4)][int(dims[1]/2)]=[0,0,0]
                    # im_data[int(dims[0]*3/4)+1][int(dims[1]/2)]=[510,510,510]
                    # im_data[int(dims[0]*3/4)-1][int(dims[1]/2)]=[510,510,510]
                    # im_data[int(dims[0]*3/4)][int(dims[1]/2)+1]=[510,510,510]
                    # im_data[int(dims[0]*3/4)][int(dims[1]/2)-1]=[510,510,510]
                    plt_img.set_data(im_data/255)
                    #add number to end of file name to prevent overwriting files
                    
                    #write image to file
                    cv2.imwrite(file_name,cv2.cvtColor(im_data, cv2.COLOR_RGB2BGR))
                if(shot_acc):
                    plt.title(str(shot_acc*100)+
                    "\np1 acc: "+str(p1_hit_count*100/p1_shot_count)[0:3]+"  kills: "+str(p2_death_count)
                    +"\np2 acc: "+str(p2_hit_count*100/p2_shot_count)[0:3]+"  kills: "+str(p1_death_count))
                else:
                    plt.title(str(0.00)+
                    "\np1 acc: "+str(p1_hit_count*100/p1_shot_count)[0:3]+"  kills: "+str(p2_death_count)
                    +"\np2 acc: "+str(p2_hit_count*100/p2_shot_count)[0:3]+"  kills: "+str(p1_death_count))
            except Exception as e:
                print("View error: ",e)
            

def headshot_inf(shared_hs):
    import torch
    import time
    import numpy as np
    model_hs = torch.hub.load("ultralytics/yolov5", 'custom', path="yolov5m_c.pt")
    print_time=time.time()
    output_results=[]
    while True:
        img=shared_hs["img"]
        try:
            tphi=time.time()
            results_hs = model_hs(img)
            output_results=[]
            for result_arr in results_hs.pred[0]:
                if (result_arr[5]==1.0):
                    output_results.append(np.array(result_arr.cpu()).copy())
            shared_hs["result"]=output_results
            if (time.time()-print_time)>3:
                print_time=time.time()
                print("hs run update- loop time:",time.time()-tphi,"prediction: ",output_results)
                results_hs.print()
        except Exception as e:
            print("HS Error: ",e)
            time.sleep(2)

if __name__=="__main__":
    sys.path.append(r"D:\vscode\mmlab1\mmdetection")
    import mmdet
    import mmcv
    from mmdet.apis import inference_detector, show_result_pyplot
    from mmdet.models import build_detector
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version
    from mmcv.runner import load_checkpoint
    import torch, torchvision
    import ultralytics
    from ultralytics import YOLO

    comm_file_namep1=r"D:\vscode\mmlab1\commp1.txt"
    comm_file_namep2=r"D:\vscode\mmlab1\commp2.txt"

   

    device="cuda"
    config_file=r"D:\vscode\mmlab1\mmdetection\configs\ms_rcnn\ms_rcnn_x101_32x4d_fpn_1x_coco.py"
    checkpoint_file=r"D:\vscode\mmlab1\mmdetection\checkpoints\ms_rcnn_x101_32x4d_fpn_1x_coco_20200206-81fd1740.pth"

    config_file=r"D:\vscode\mmlab1\mmdetection\configs\yolact\yolact_r50_1x8_coco.py"  #yolact
    checkpoint_file=r"D:\vscode\mmlab1\mmdetection\checkpoints\yolact_r50_1x8_coco_20200908-f38d58df.pth"

    #config_file=r"D:\vscode\mmlab1\mmdetection\configs\instaboost\mask_rcnn_r50_fpn_instaboost_4x_coco.py" #instaboost
    #checkpoint_file=r"D:\vscode\mmlab1\mmdetection\checkpoints\mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-d025f83a.pth"

    #print if config file is loaded
    # print(os.path.exists(config_file))

    # #setup model
    # config = mmcv.Config.fromfile(config_file)
    # config.model.pretrained = None
    # model = build_detector(config.model)
    # checkpoint = load_checkpoint(model, checkpoint_file, map_location=device)
    # model.CLASSES = checkpoint['meta']['CLASSES']
    # model.cfg = config
    # model.to(device)
    # model.eval()




    run_val=True

    #player settings
    p1_health=100
    p2_health=100
    #damage_mult_b=30
    damage_headshot=100
    damage_mult_b=24
    damage_mult=damage_mult_b


    file_que=[]
    
    p1_ims_list=os.listdir(p1_ims_dir)
    p2_ims_list=os.listdir(p2_ims_dir)
    p1_ims_list_prev=p1_ims_list
    p2_ims_list_prev=p2_ims_list


    slow_print_time=time.time()

    p1_dead=False
    p2_dead=False
    p1_dead_prev=False
    p2_dead_prev=False

    p1_acc=100#player stats setup
    p2_acc=100
    p1_shot_count=0.01
    p2_shot_count=0.01
    p1_hit_count=0
    p2_hit_count=0
    p1_death_count=0
    p2_death_count=0

    p1_dead_time=time.time()
    p2_dead_time=time.time()
    max_h=0
    min_h=0



    warm_time=time.time()
    t_p_i=time.time()

    l_t=time.time()
    t_u=time.time()
    respawn_time=3  #respawn time


    manager = multiprocessing.Manager()#data share
    shared_data=manager.dict()
    shared_hs=manager.dict()
    shared_hs["img"]=""
    shared_hs["result"]=[]
    shared_data["acc"]=0
    shared_data["mask"]=[]
    shared_data["img"]=[]
    shared_data["headshot"]=False
    shared_data["result"]=[]
    shared_data["read"]=0
    shared_data["stats_p1"]=[p1_shot_count,p1_hit_count,p1_death_count]#shots hits deaths
    shared_data["stats_p2"]=[p2_shot_count,p2_hit_count,p2_death_count]
    shared_data["p1_hit_count"]=0
    shared_data["p2_hit_count"]=0

    shared_que_info=manager.dict()
    shared_que_info["file_que_new"]=[]
    shared_que_info["signal"]=0
    shared_que_thread=multiprocessing.Process(target=file_que_handler, args=(shared_que_info,))
    shared_que_thread.start()

    view_thread = multiprocessing.Process(target=view_handler, args=(shared_data,))
    #headshot_thread = multiprocessing.Process(target=headshot_inf, args=(shared_hs,))
    view_thread.start()
    #headshot_thread.start()

    respawns=True
    update_que=True
    if headshots_on:
        model_hs = torch.hub.load("ultralytics/yolov5", 'custom', path="yolov5m_c.pt")
    model_main = YOLO('yolov8n-seg.pt')
    #img=model_main.predict('el1_ims/incoming_img'+str(1)+'.JPG')#,save=True)
    #print(img)
    # while True:
    #     time_new_model=time.time()
    #     img=model_main.predict('el1_ims/incoming_img'+str(1)+'.JPG')
    #     print(time.time()-time_new_model)
    while run_val:
        top_while_time=time.time()
        t11=time.time()
        gun_info_file=open("gun_info.txt",'r')#get gun details
        p1_gun_details=gun_info_file.readline()
        p2_gun_details=gun_info_file.readline()
        gun_info_file.close()
        damage_mult_b_p1=int(p1_gun_details.split(',')[4])
        damage_mult_b_p2=int(p2_gun_details.split(',')[4])
        damage_headshot_p1=int(p1_gun_details.split(',')[5])
        damage_headshot_p2=int(p2_gun_details.split(',')[5])

        

    
        if update_que:#file que updates
            file_que=file_que+shared_que_info["file_que_new"]
            time.sleep(0.001)
            if len(file_que)>0:
                if shared_que_info["signal"]>5:
                    shared_que_info["signal"]=0
                else:
                    shared_que_info["signal"]+=1
            shared_data["stats_p1"]=[p1_shot_count,p1_hit_count,p1_death_count]#shots hits deaths
            shared_data["stats_p2"]=[p2_shot_count,p2_hit_count,p2_death_count]
        tf=time.time()


        if ((time.time()-slow_print_time)>2):#slow prints
            slow_print_time=time.time()
            print(p1_health,p2_health,file_que, "warm:",warm_time, " loop T: ",time.time()-l_t)
        l_t=time.time()


        if respawns:#respawn logic
            if p1_health<=0:#p1 dead respawn
                p1_dead=True
                if (time.time()-p1_dead_time)>respawn_time:
                    p1_dead_prev=False
                    p1_health=100
                    comm_filep1=open(comm_file_namep1,'w')
                    comm_filep1.write("0,"+str(p1_health)+p1_gun_details)
                    comm_filep1.close()
                    p1_dead=False
            else:
                p1_dead_time=time.time()
                p1_dead=False
            if p2_health<=0:#p2 respawn
                p2_dead=True
                if (time.time()-p2_dead_time)>respawn_time:
                    p2_dead_prev=False
                    p2_health=100
                    comm_filep2=open(comm_file_namep2,'r+')
                    comm_filep2.write("0,"+str(p2_health)+p2_gun_details)
                    comm_filep2.close()
                    p2_dead=False
            else:
                p2_dead_time=time.time()
                p2_dead=False
            t2=time.time()

        tpfq=time.time()
        if(len(file_que)!=0):#if a file is in file que
            if len(file_que)>10:
                file_que=file_que[(len(file_que)-9):len(file_que)]
            img=file_que[0]
            shared_data["img"]=img
            shared_hs["img"]=img

            print("shot fired img:",img)
            time.sleep(0.003)



            if(len(img)>0):#determine who is shooting
                if(img[19]=='1'):
                    p1_shooting=True
                else:
                    p1_shooting=False

            if (len(file_que)>1):#remove current file from que
                file_que=file_que[1:]
            else:
                file_que=[]

            if p1_shooting:
                p1_shot_count+=1
            else:
                p2_shot_count+=1

            try:#rescaling option
                pass
                #r_im=cv2.imread(img,cv2.IMREAD_COLOR)
                #rescale_f=3
                #res_size=[int(r_im.shape[1]/rescale_f),int(r_im.shape[0]/rescale_f)]
                #res_im=cv2.resize(r_im,res_size)
                #cv2.imwrite(img,res_im)
            except:
                pass
            if (p1_shooting and not p1_dead)or(not p1_shooting and not p2_dead):
                try:
                    prev_to_inf_time=time.time()
                    result_main=model_main.predict(img)#get inference from model
                    print("inf time: ",time.time()-prev_to_inf_time)
                    pmmt=time.time()
                    try:
                        masks_generated=[]
                        mask_generated=torch.zeros(result_main[0].masks[0].data[0].cpu().shape)
                        for ind in range(len(result_main[0].masks.cpu())):
                            it_mask=result_main[0].masks[ind].data[0].cpu()
                            it_box=result_main[0].boxes[ind].data[0].cpu()
                            #print(it_mask)
                            if it_box[5]==0.0:#only humans
                                if (it_box[4]>0.3):#accuracy threshold
                                    shared_data["acc"]=it_box[4]
                                    masks_generated.append(it_mask)
                                    mask_generated+=it_mask
                        mask_to_save=mask_generated.repeat(3,1,1)
                        shared_data["mask"]=mask_generated
                        mmt=time.time()
                    except:
                        print("found nothing")
                        shared_data["mask"]=[]
                        shared_data["acc"]=0
                    if (len(masks_generated)>0):#if there are humans
                        dims=mask_generated.shape
                        valid_hit=0.0
                        valid_hit+=(mask_generated[int(dims[0]*3/4)][int(dims[1]/2)])
                        valid_hit+=(mask_generated[int(dims[0]/4)][int(dims[1]/2)])


                        if valid_hit>0.0:#if there is a valid hit
                            if p1_shooting:
                                damage_mult=damage_mult_b_p1
                            else:
                                damage_mult=damage_mult_b_p2
                            t_p_i=time.time()
                            shared_data["headshot"]=False
                            if headshots_on:
                                tph=time.time()
                                results_hs = model_hs(img)#get headshot prediction
                                print("hs pred: ",results_hs.pred[0])
                                hs_res_arr=[]
                                for hs_arr in results_hs.pred[0]:
                                    if hs_arr[5]==1.0:
                                        hs_res_arr.append([hs_arr[0],hs_arr[1],hs_arr[2],hs_arr[3],hs_arr[4]])
                                for hs_res in hs_res_arr:
                                    if (hs_res[4]>0.3):
                                        if ((hs_res[0]<160) and (hs_res[2]>160)):
                                            if ((hs_res[1]<120) and (hs_res[3]>120)):
                                                if p1_shooting:
                                                    damage_mult=damage_headshot_p1
                                                else:
                                                    damage_mult=damage_headshot_p2
                                                shared_data["headshot"]=True
                                            elif ((hs_res[1]<360) and (hs_res[3]>360)):
                                                if p1_shooting:
                                                    damage_mult=int(damage_headshot_p1*0.9)
                                                else:
                                                    damage_mult=int(damage_headshot_p2*0.9)
                                                shared_data["headshot"]=True
                                print(time.time()-tph)
                                        

                            if p1_shooting:#if player 1 shot player 2
                                p2_health-=damage_mult
                                if p2_health<=0:
                                    p2_dead=True
                                    p2_health=0
                                p1_hit_count+=1
                                shared_data["p1_hit_count"]=p1_hit_count
                                comm_filep1=open(comm_file_namep1,'w')
                                if p2_dead:
                                    if not p2_dead_prev:
                                        comm_stringp1="2,"+str(p1_health)
                                        p2_death_count+=1
                                        p2_dead_prev=True
                                    else:
                                        comm_stringp1="0,"+str(p1_health)
                                else:
                                    comm_stringp1="1,"+str(p1_health)
                                comm_filep1.write(comm_stringp1+p1_gun_details)#give hit marker
                                comm_filep1.close()

                                comm_filep2=open(comm_file_namep2,'w')
                                comm_filep2.write("0,"+str(p2_health)+p2_gun_details)
                                comm_filep2.close()
                            else:#player 2 shot player 1
                                p2_hit_count+=1
                                shared_data["p2_hit_count"]=p2_hit_count
                                comm_filep2=open(comm_file_namep2,'w')
                                p1_health-=damage_mult
                                if p1_health<=0:
                                    p1_health=0
                                    p1_dead=True
                                if p1_dead:
                                    if not p1_dead_prev:
                                        comm_stringp2="2,"+str(p2_health)
                                        p1_death_count+=1
                                        p1_dead_prev=True
                                    else:
                                        comm_stringp2="0,"+str(p2_health)
                                else:
                                    comm_stringp2="1,"+str(p2_health)#give hit marker
                                comm_filep2.write(comm_stringp2+p2_gun_details)
                                comm_filep2.close()

                                comm_filep1=open(comm_file_namep1,'w')
                                comm_stringp1="0,"+str(p1_health)#write health
                                comm_filep1.write(comm_stringp1+p1_gun_details)
                                comm_filep1.close()
                    shared_data["stats_p1"]=[p1_shot_count,p1_hit_count,p1_death_count]#shots hits deaths
                    shared_data["stats_p2"]=[p2_shot_count,p2_hit_count,p2_death_count]
                    shared_data["read"]+=1
                    #print("full step time: ",time.time()-tpfq,mmt-pmmt," full loop: ",time.time()-top_while_time)
                    print("Full loop: ",time.time()-top_while_time,"\ndetection: ",time.time()-tpfq,"\nupdate: ",tf-top_while_time, "\nrespawn: ",tpfq-tf)
                except Exception as e:
                    t, o, b=sys.exc_info()
                    print("poop ",e, b.tb_lineno)#result[0][0])
                    print("\n\n\n\n\n\n\nn\n\n\n\n\nn\nn\n\n\n\n\n\n\n")



        if (time.time()-t_p_i)>1:#warmer logic
            try:
                warm_time=time.time()
                if headshots_on:
                    model_hs(img)#"keep_warm.jpg")
                
                model_main.predict(img)#"keep_warm.jpg")
                warm_time=time.time()-warm_time
                print("warm: ",warm_time)
                t_p_i=time.time()
            except:
                pass
