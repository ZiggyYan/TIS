# TIS
## Code Explaination
1. Dataset - The place you put your dataset. <br>
2. code - The main code of our structure. <br>
3. side - Some supportive external codes. <br>
### Inside code
1. core - The main part of our structure. <br>
2. GANs - Generator and Discriminator Source Code. Here we provide multiple generator and discriminator architectures for selection. Through experimentation, we have identified the context_encoder as the optimal choice that best meets our requirements. <br>
#### Inside core
1. train.py - The part for training the sturcture. <br>
2. infer.py - The part for testing our codes on data. <br>
3. datasets.py - The part for dataset specifications. <br>
4. discriminators - The part for backbone discriminator. <br>
5. sides - The part for HOG Loss and Block Loss. <br>
#### Inside GANs
1. implementations - The source codes for GANs. Among GANs, context_encoder is the one we used in our work. <br>
     Inside context_encoder: <br>
   a) datasets.py - The part for dataset specifications in pretraining GANs. <br>
   b) models.py - The source code of context_encoder. <br>
   c) phase1_context_encoder.py - Code for phase 1 training in GANs. <br>
   d) phase2-1_discriminator.py - Code for phase 2-1 training in GANs. <br>
   e) phase2-2_discriminator.py - Code for phase 2-2 training in GANs. <br>
   f) phase2extra_discriminator.py - This code evaluates whether the discriminator's judgment capability on the original dataset degrades after Phase 2-2. Experimental results confirm performance deterioration (Note: Not used for pretraining).
   g) phase22datasets.py - Special designed code of datasets for phase 2-2 training in GANs. <br>
   h) test_models - The part we used for testing if other GANs can perform better with the structure of context_encoder. The answer is NO. <br>
3. model_weights - Pretrained weights used in our work.
## Get Access to Our Work
Everything is available on Baidu Cloud for download as needed if you dont want to download from github:
### Dataset
We used two datasets: the full BUSB collection and just the healthy images from BUSI. Download them using the link below and replace everything in your 'Dataset' folder. <br>
Link: https://pan.baidu.com/s/1Gvd6OGgKdiOHt_ghiPWyQQ?pwd=3gkq Password: 3gkq <br>
### Outputs
All our experimental results - including architecture .pth files and output images - are available in the link below. Feel free to download them for direct use. <br>
Link: https://pan.baidu.com/s/1Bdz2mhMUXoLD8lmaiiuDGw?pwd=wbiv Password: wbiv <br>
### Pure Codes
This section contains the core model architecture source code, which excludes both datasets and output files. The content is similar to what we've uploaded on GitHub. <br>
Link: https://pan.baidu.com/s/12ceTF7mUf9fFnL-W4qCD8w?pwd=2vv8 Password: 2vv8  <br>
### Complete TIS
This version contains the full dataset and all corresponding output files, pre-organized in their operational directory structure. By downloading this package, you can immediately execute our code without additional setup. <br>
Link: https://pan.baidu.com/s/1EFhJGJFblf7bffyWNtIg-w?pwd=pmw1 Password: pmw1 
### SOTAs experiments
Here we provide the code used in the SOTA experiments from the paper, making it easy for anyone who needs to reproduce the results. All the code has been modified—just tweak a few local settings and you're good to go. The link below will auto-populate the access password. <br>
·Codes:<br>
1. ReDO <br>
Link: https://pan.baidu.com/s/1bF6jEWRuQTPewh8JfcdGOA?pwd=1xnp Password: 1xnp <br>
2. STEGO <br>
Link: https://pan.baidu.com/s/1TY2PDEx8gw9kQMSlRG-EFw?pwd=wed6 Password: wed6 <br>
3. SmooSeg <br>
Link: https://pan.baidu.com/s/1XqteJsIsWKtmf2GoE4bXQA?pwd=aq6g Password: aq6g <br>
4. deep_spectral_segmentation <br>
Link: https://pan.baidu.com/s/1F8srb5nLcVzDILXMvJt_ZQ?pwd=wbf5 Password: wbf5 <br>
5. Labels4Free <br>
Link: https://pan.baidu.com/s/1u8Oai977-XCtCJqKwNA1Ng?pwd=htd3 Password: htd3 <br>
·Datasets: <br>
1. deep_spectral_segmentation <br>
Link: https://pan.baidu.com/s/1p0rG-6ywUiVlEMs-N2kMjQ?pwd=24f4 Password: 24f4 <br>
2. SmooSeg|STEGO <br>
Link: https://pan.baidu.com/s/1_mlUZFSU0HzOvRQ9T_opYQ?pwd=8fax Password: 8fax <br>
3. ReDO|Labels4Free<br>
Link: https://pan.baidu.com/s/1ZHgLwRs-4fTQzH9BNIv6kQ?pwd=ft3y Password: ft3y <br>

## Closing Remarks
If you have any questions regarding the file download or the article, feel free to contact me at: 231020050@fzu.edu.cn or 506264025@qq.com
