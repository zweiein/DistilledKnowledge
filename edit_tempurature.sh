. ./cmd.sh
. ./path.sh

stage=4

#/usr/local/kaldi-trunk/src/nnetbin/nnet-copy --binary=false /share/homes/yang/timit/s6/exp/edit_tempuature/nnet.init /share/homes/yang/timit/s6/exp/edit_tempuature/nnet.nobinary.init

# --proto-opts "--block-softmax-dims=3400:144 --block-softmax-tempurature=11:21"
# /share/homes/yang/timit/s6/exp/edit_tempurature_not_fix_bp_yet  --> 這個是還沒改softmax加溫的back-propagation，而且monophone用hard label的結果

if [ $stage -le 3 ]; then
steps/nnet/train.sh \
  --delta-opts --delta-order=2 --splice 5 \
  --labels scp:/share/homes/yang/timit/s6/exp/mtl_dnn4_tri3mono_1-1/pasted_post.scp \
  --num-tgt 3544 \
  --proto-opts "--block-softmax-dims=3400:144 --block-softmax-tempurature=1:20" \
  --train-tool "nnet-train-frmshuff --objective-function=multitask,xent,3400,1,1,xent,144,1,20" \
  --learn-rate 0.008 --hid-layers 4 \
  /share/homes/yang/timit/s6/data-mfcc-mtl/combine_all_tr /share/homes/yang/timit/s6/data-mfcc-mtl/combine_all_cv lang-dummy ali-dummy ali-dummy /share/homes/yang/timit/s6/exp/edit_tempuature 
fi

#echo "---Model OK!---"
#exit;

dir=/share/homes/yang/timit/s6/exp/edit_tempuature
ali1_dir=exp/tri3_combine_ali

gmm=exp/tri3_combine
 
if [ $stage -le 4 ]; then
   ali1_pdf="ark:ali-to-pdf ${gmm}_ali/final.mdl 'ark:gunzip -c ${gmm}_ali/ali.*.gz |' ark:- |"
   
   # Create files used in decdoing, missing due to --labels use,
   cp /share/homes/yang/timit/s6/exp/mtl_dnn4_tri3mono_1-1/ali_train_pdf.counts  /share/homes/yang/timit/s6/exp/edit_tempuature
   copy-transition-model --binary=false $ali1_dir/final.mdl $dir/final.mdl
   cp $ali1_dir/tree $dir/tree

   # Rebuild network, <BlockSoftmax> is removed, and neurons from 1st block are selected,
   nnet-concat "nnet-copy --remove-last-components=1 $dir/final.nnet - |" \
    "echo '<Copy> <InputDim> 3544 <OutputDim> 3400 <BuildVector> 1:3400 </BuildVector>' | nnet-initialize - - |" \
    $dir/final.nnet.lang1
 fi

lang_dir=graph_tgpr_lang_test

if [ $stage -le 5 ]; then
       utils/mkgraph.sh /share/homes/yang/timit/s6/data/lang_test  $dir  $dir/$lang_dir
fi

if [ $stage -le 6 ]; then
  # Decode (reuse HCLG graph)
    lang_decode=decode_graph_tgpr_lang_test

    steps/nnet/decode.sh --nj 4  --acwt 0.10 --config /share/homes/yang/timit/s6/conf/decode_dnn.config \
		  --nnet $dir/final.nnet.lang1 --nnet-forward-opts "--no-softmax=true --prior-scale=1.0" \
      $dir/$lang_dir  /share/homes/yang/timit/s6/data-mfcc-mtl/test  $dir/$lang_decode
    
fi


if [ $stage -le 8 ]; then
    # Display %PER
	  lang_decode=decode_graph_tgpr_lang_test
    for x in $dir/$lang_decode; do [ -d $x ] && echo $x | grep "${1:-.*}" && grep WER $x/wer_*  | utils/best_wer.sh; done
    for x in $dir/$lang_decode; do [ -d $x ] && echo $x | grep "${1:-.*}" && grep WER $x/cer_*  | utils/best_wer.sh; done
fi


:<<BLOCK

steps/nnet/train.sh \
  --delta-opts --delta-order=2 --splice 5 \
  --labels scp:/share/homes/yang/timit/s6/exp/mtl_dnn4_tri3mono_1-1/pasted_post.scp \
  --num-tgt 3544 \
  --proto-opts "--block-softmax-dims=3400:144 --block-softmax-tempurature=1:1" \
  --train-tool "nnet-train-frmshuff --objective-function=multitask,xent,3400,1,1,xent,144,1,1" \
  --learn-rate 0.008 --hid-layers 4 \
  /share/homes/yang/timit/s6/data-mfcc-mtl/combine_all_tr /share/homes/yang/timit/s6/data-mfcc-mtl/combine_all_cv lang-dummy ali-dummy ali-dummy /share/homes/yang/timit/s6/exp/edit_dnn_tempuature 


steps/nnet/train.sh \
  --cmvn-opts "--norm-means=true --norm-vars=true" --delta-opts --delta-order=2 --splice 5 \
  --learn-rate 0.008 --hid-layers 4 \
  /share/homes/yang/timit/s6/data_fbank/train /share/homes/yang/timit/s6/data_fbank/dev /share/homes/yang/timit/s6/data/lang  /share/homes/yang/timit/s6/exp/tri3_combine_ali /share/homes/yang/timit/s6/exp/tri3_combine_ali /share/homes/yang/timit/s6/exp/edit_dnn_tempuature 

steps/nnet/train.sh \
  --cmvn-opts "--norm-means=true --norm-vars=true" --delta-opts --delta-order=2 --splice 5 \
  --labels scp:/share/homes/yang/timit/s6/exp/mtl_dnn4_tri3mono_1-1/pasted_post.scp \
  --num-tgt 3544 \
  --proto-opts --block-softmax-dims='3400:144' \
  --train-tool "nnet-train-frmshuff --objective-function=multitask,xent,3400,1,xent,144,1" \
  --learn-rate 0.008 --hid-layers 4 \
  /share/homes/yang/timit/s6/data-mfcc-mtl/combine_all_tr /share/homes/yang/timit/s6/data-mfcc-mtl/combine_all_cv lang-dummy ali-dummy ali-dummy /share/homes/yang/timit/s6/exp/edit_tempuature 
BLOCK