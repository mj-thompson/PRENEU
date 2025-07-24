import tensorflow as tf
import PRENEU as prn
import PRENEU_models as prnmod
import numpy as np
import pandas as pd
import sys
from pathlib import Path
curarg=int(sys.argv[1])-1
curarg=str(curarg)

h2=0.80
alpha=0.5
psi=0.0
N=300000
delta=0

def get_corr(pred,true):
    bs_auc=[]
    rngseed=np.random.RandomState(1999)
    for i in range(1000):
        ### operate on the 480k first
        indx=rngseed.randint(0,len(pred),len(pred))
        if len(np.unique(true[indx])) < 2:
            continue
        bs_auc.append(np.corrcoef(true[indx], pred[indx])[1,0])

    sortedauc = np.array(bs_auc)
    sortedauc.sort()
    aucobs=np.corrcoef(true,pred)[1,0]
    auctup=(sortedauc[25],aucobs,sortedauc[975])
    print("p corr: " + str(round(sortedauc[25],5)) + ","+\
          str(round(aucobs,5))+ ","+\
          str(round(sortedauc[975],5)))
    return auctup 
# early stopping callback

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=10, 
                                            verbose=1, 
                                            mode='min', 
                                            restore_best_weights=True)
# reduce learning rate callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                factor=0.2,
                                                patience=6, 
                                                min_lr=1e-6,
                                                mode='min',
                                                verbose=1) 

savedir="/u/project/halperin/mjthomps/motif_sims/res_fis/"
import os
import shutil
modlist=["cnn", "can", "dilcnn_resid", "dilcnn_noresid", "lstm_seq", "lstm_final", "lstm_seq_and_final"]
uppers=[]
lowers=[]
scores=[]
preds=[]
modsdone=[]
batch_size=100
epochs=100


## all the other parameters...
savesuffix=prn.get_save_suffix(curargument=curarg,
                                                h2=h2,
                                                alpha=alpha,
                                                psi=psi,
                                                N=N,
                                                delta=delta)
filepath="/u/project/halperin/mjthomps/motif_sims/nn_data/data"+savesuffix+".pickle"
checkpoint_root = Path("/u/project/halperin/mjthomps/motif_sims/checkpoints")

if not Path(filepath).exists():
    Xtr, Ytr, Xte, Yte, savesuffix = prn.simulate_data(curargument=curarg,
                                                h2=h2,
                                                alpha=alpha,
                                                psi=psi,
                                                N=N,
                                                delta=delta)
else:
    import pickle
    with open(filepath, "rb") as f:
        Xtr, Ytr, Xte, Yte= pickle.load(f)

import shutil


for curmodtype in modlist:
    curmod, curname=prnmod.get_model(modtype=curmodtype,savesuffix=savesuffix)
    if Path(curname).exists():
        curmod.load_weights(curname)
    else:
        optimizer=curmod.optimizer
        curcheckptname=curname.split("/")[7]
        model_ckpt_dir = checkpoint_root / curcheckptname
        os.makedirs(model_ckpt_dir, exist_ok=True)
        epoch_var = tf.Variable(0, name="epoch_var", dtype=tf.int64)
        ckpt = tf.train.Checkpoint(
            epoch=epoch_var,
            model=curmod,
            optimizer=optimizer,
        )
        ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            directory=model_ckpt_dir,
            max_to_keep=1
        )
        start_epoch = 0
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            start_epoch = int(epoch_var.numpy())
            print(f"Restored checkpoint from epoch {start_epoch}, LR = {optimizer.lr.numpy():.2e}")
        class SaveEpochCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                # Update our epoch variable (so it’s stored in the checkpoint)
                epoch_var.assign(epoch + 1)
                # Save checkpoint
                ckpt_path = ckpt_manager.save()
                print(f"\n Saved checkpoint for epoch {epoch+1}: {ckpt_path}")
        history=curmod.fit(x=Xtr[0],y=Ytr,
                        batch_size=batch_size,epochs=epochs,verbose=1,
                        initial_epoch=start_epoch,
                        validation_split=0.10,
                        callbacks=[es_callback, reduce_lr,SaveEpochCallback()])
        curmod.save_weights(curname) 
        shutil.rmtree(model_ckpt_dir)


    tmp=curmod.predict(Xte[0])
    curscore=get_corr(tmp.flatten(),Yte)
    # Try again if didn't converge
    # if np.isnan(curscore[0]):
    #     curmod, curname=prnmod.get_model(modtype=curmodtype,savesuffix=savesuffix)
    #     history=curmod.fit(x=Xtr[0],y=Ytr,
    #                     batch_size=batch_size,epochs=epochs,verbose=1,
    #                     validation_split=0.10,
    #                     callbacks=[es_callback, reduce_lr])
    #     tmp=curmod.predict(Xte[0])
    #     curscore=get_corr(tmp.flatten(),Yte)
 
    
    
    uppers.append(curscore[2])
    scores.append(curscore[1])
    lowers.append(curscore[0])
    preds.append(tmp)
    modsdone.append(curmodtype)
        
    scoredf=pd.DataFrame({"modtype" : modsdone,
                          "corr" : scores,
                          "upper" : uppers,
                          "lower" : lowers})
    scoredf["runno"]=curarg
    preddf=pd.DataFrame(np.array(preds).squeeze().T, columns=modsdone)
    preddf["true"]=Yte

    scoredf.to_csv(savedir+"scores"+savesuffix+".tsv",sep="\t",index=False)
    preddf.to_csv(savedir+"preds"+savesuffix+".tsv",sep="\t",index=False)

