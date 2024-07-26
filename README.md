# SegGPT Fine-Tune  
## Fine Tuning Method
0. **환경 설정**  
```bash
conda env create -f segGPT.yml
```
1. **Data format**  
```
./dataset/train_dataset
                ├── train
                │   ├── all image
                └── val
                    ├── all label (which has same name with images)

```  
이렇게 만들기 위해서 **transform_data_to_one.ipynb** 파일을 사용했습니다.

2. **Code correction log**  

``` python
def _lbl_random_color(self, label: np.ndarray, color_palette: np.ndarray):
        result = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

        # for i in range(self.n_classes):
        #     result[label==i] = color_palette[i]
        result[label==0] = color_palette[0]
        result[label!=0] = color_palette[1]
        
        return result
```  
정답 레이블을 두가지 랜덤 색상으로 바꾸어줍니다. 검은색과 흰색으로 설정할 경우, 학습이 정상적으로 진행되지 않고, 스텝이 진행될 수록 예측값이 검은 색만 나오게되었습니다. 따라서 이렇게 랜덤하게 색상을 계속 바꿔가면서 학습을 하여야 스텝이 지나도 정상적으로 예측이 진행됩니다.

3. **전반적인 학습과정**  
* **Data 변형 : Input Data Format**  
H = 488, W = 488  
이미지 2개가 위아래로 합쳐져 (Batch size, 3, H*2, W) 형태로 입력됩니다. 정답 레이블도 마찬가지입니다.  
이때 정답 레이블은 위에서 언급한대로, 랜덤 색상 2개를 뽑아, 흰색 영역과 검은색 영역이던 원래의 레이블을 다른 색상으로 채우게됩니다.  
그리고 랜덤한 위치에 마스크를 씌우게 됩니다. config/Base.json에서 발견되는 mask_ratio 옵션은 이 마스크의 비율을 결정합니다.  

* **Agent**  
전반적인 학습과정은 Agent 클래스가 담당합니다.  
1. do_training으로 학습시작
2. step메서드로 모델로부터 예측값을 받음. (loss는 모델 안에서 동시에 계산됨.)
3. 만들어진 pred가 iou 메서드로 들어가, iou를 계산함과 동시에 cmap_to_lbl 함수를 이용해 기존에 pred를 l2거리를 계산해 레이블 이미지를 생성한다.  
이미지가 16x16패치로 쪼개진 상태로 각각 ViT로 들어가 나온 예측값인만큼, 각 패치 경계도 보이고, 흐릿한 이미지를 가지는데 이러한 예측을 깔끔하게 만들어줌.  

4. 학습 시작 명령
```bash
python train.py --port {안쓰는 port 입력}
```
5. inference 시작 명령
```bash
python inference.py --model-path /shared/home/vclp/hyunwook/junhyung/segGPT_origin/SegGPT-FineTune/seggpt_vit_large.pth --prompt-img-dir /shared/home/vclp/hyunwook/junhyung/segGPT_origin/SegGPT-FineTune/dataset/train_dataset/train/images --prompt-label-dir /shared/home/vclp/hyunwook/junhyung/segGPT_origin/SegGPT-FineTune/dataset/train_dataset/train/labels --dataset-dir /shared/home/vclp/hyunwook/junhyung/segGPT_origin/SegGPT-FineTune/dataset/train_dataset/val/images --mapping /shared/home/vclp/hyunwook/junhyung/segGPT_origin/SegGPT-FineTune/mappings/mapping_vit_filtered.json --top-k 1 --outdir /shared/home/vclp/hyunwook/junhyung/segGPT_origin/SegGPT-FineTune/output --split 1
```

