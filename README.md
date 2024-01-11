# 1. 데이터 수집

http://www.cs.columbia.edu/CAVE/databases/pubfig/

**`PubFig: Public Figures Face Database`에서 제공하는 eval_urls.txt를 이용하여 HTTP Client를 이용해서 사람 사진 수집**

```
|-Aaron Eckhart
 |-Adam Sandler
 |-Adriana Lima
 |-Alberto Gonzales
 |-Alec Baldwin
 |-Alicia Keys
 |-Angela Merkel
 |-Angelina Jolie
 |-Anna Kournikova
 |-Antonio Banderas
 |-Ashley Judd
 |-Ashton Kutcher
 |-Avril Lavigne
 |-Ben Affleck
 |-Beyonce Knowles
 |-Bill Clinton
...
 |-Gordon Brown
 |-Gwyneth Paltrow
 |-Halle Berry
 |-Harrison Ford
 |-Holly Hunter
 |-Hugh Grant
 |-Jack Nicholson
 |-James Franco
 |-James Gandolfini
 |-Jason Statham
 |-Javier Bardem
 |-Jay Leno
 |-Jeff Bridges
 |-Jennifer Aniston

# 65명, 한 사람당 약 30장
```

# 2. GCP Vision API를 이용하여 crop

학습을 위해선 정상/비정상 사진을 가려야 할 필요가 있음.

labeling도 필요함 → 폴더 이름(사람이름)기준으로 나눈 integer로 labeling

```
desc/training/brad-pitt028.jpg,Brad Pitt,1
desc/training/williambradleypitt.jpg,Brad Pitt,1
desc/training/intyou1.jpg,Brad Pitt,1
...
```
### 크롤링한 이미지의 품질이 안 좋기 때문에 구글의 Vision API를 사용함.
1. 최대한 정면 사진일 것. (각도 조절 가능)
2. 한 사람만 나오는 사진일 것.
-> 손상된 이미지가 많고 각 사람별로 사진 개수가 다름.

# 3-1. CNN 아키텍처를 이용해서 모델 설계

1. 이 아키텍처는 세 개의 합성곱 레이어와 풀링 레이어로 이루어짐. 각 합성곱 레이어 다음에는 ReLU 활성화 함수 사용.
2. 합성곱과 풀링 레이어를 통해 입력 이미지의 특징을 추출하고, 세 번의 풀링 작업 후 이미지 크기가 12x12로 축소.
    
    (96 → 48 → 24 → 12)
    
3. 마지막으로, 전체 연결 레이어 두 개를 통해 이미지의 특징을 분류하며, 중간에 Dropout 레이어를 추가하여 과적 방지.
