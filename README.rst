# Lessons learned

1. Depthwise Convolutions are quite slow in PyTorch. On my GFX 1080, EfficientNet-B4 on 384x384 images takes 3 days to generate OOF predictions (5-fold CV). This significantly limits the ability to iterate fast and the number of experiments one can run. Maybe its time to buy a second GPU so that I can run more than one experiment at a time.
2. Kaggle Notebooks were quite unstable for me. Often they crashed without any apparent reason. Importing data in Google Colab takes ages. Once data is imported in Google Drive, I wasn't able to successfully run a model due to IO Errors. Apparently this is related to reading many image files instead of one single file ― e.g., a TFRecord.
3. The main lessson from this competition was that I need to have a decent setup in Colab/Kaggle Notebook so that I can leverage the free resources to complement my physical rig. I'll invest time in my next competition to ensure that is the case. A physical deep learning server will always be better IMHO, but you can't afford to not leverage free resources to competite with the very best.
4. This was my first time using PyTorch. PyTorch-Lightning is truly nice.
5. This repo contains a lot of duplicate code. I had very little time to dedicate to this competition so opted for copy-pasting and litte refactoring to move fast. The code is quite a mess. I'm planning to move the important bits into SleepMind so that I can easily reuse it in future competitions.
6. Once again, Public LB is not reflective of Private LB results. Trust your CV!
7. I spent a significant amount of time tuning some parameters here and there. This is often a faste of time that could be used to experiment with different methods ― e.g., pseudo labeling, images of larger resolution.
8. LRFinder rocks! ― although it doesn't really work that well with Over9000...
9. I should have spent the last two weeks training EfficientNet models on larger resoutions. The main model I have was quite good but, differently from the top teams, I never used images with resolution higher than 384x384. What a mistake! A top 10% finish was within reach.
