---
title: "[논문리뷰] Diffusion Models already have a Semantic Latent Space"
subtitle: 
categories: Review
date: 2023-12-01 02:50:43 +0900
tags:
  - difussion model
  - h-space
  - image editing
  - asyrp
  - gan
  - generative ai
  - image generating
toc: true
toc_sticky: true
---

# 💡 핵심 요약
1. 기존 diffusion 모델에서 생성 과정을 제어할 때 발생했었던 문제를 해결하는 방법 제안
    - Asyrp을 제안하여 중간 변화가 상쇄되는 문제를 해결

2. Diffusion 모델에서 이미지 생성 과정을 제어할 수 있는 semantic latent space인 h-space의 발견 
   - GAN에서 latent space를 편집해서 이미지를 제어하는 것과 같이 diffusion 모델에서도 같은 방식으로 이미지를 제어할 수 있음)

3. 생성 과정의 프로세스를 Asyrp을 이용한 editing, 기존 denoising, quality boosting의 3단계로 나눠서 다양한 이미지를 생성하고 좋은 품질의 이미지를 생성하도록 함



# **Introduction**

기존 diffusion 모델에서 이미지 생성을 제어하는 과정에 대한 설명이다. 

![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled.png)

(a) Image guidance는 unconditional한 latent variable에 guiding image의 latent variable을 합치는 방식을 사용한다. 하지만, 가이드와 unconditional 결과 중에 어떤 속성을 반영할지 지정하는 것이 모호하고, 변화의 크기에 대한 직관적인 제어가 부족하다는 문제점이 있다. 

(b) Classifier guidance는 diffusion model에 classifier를 추가하여 목표 클래스와 일치하도록 reverse process에서 latent variable에 classifier의 기울기를 부과하여 이미지를 조작한다. 이 방법은 classifier를 추가로 훈련해야 하고, 샘플링 중에 classifier를 통해 기울기를 계산하는 비용이 크다는 문제점이 있다. 

본 논문에서는 frozen diffusion model의 semantic latent space를 발견하는 비대칭 역방향 프로세스(Asyrp)를 제안한다. 그렇게 해서 발견한 semantic latent space를 h-space라고 칭한다. 본 논문에서는 사전 훈련된 frozen diffusion model에서 semantic latent space를 최초로 발견하였다.

# 2. Background

Sematnic latent space에 대해 이야기 하기 전에 DDIM의 reverse process 식을 살펴보는 것으로 시작한다. DDIM에서는 non-Markovian process를 이용해서,  DDPM의 forward process식을 다음과 같이 재정의한다. 

## DDPM, DDIM

- DDPM의 forward process

![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_1.png)

- DDIM의 forward process

$$
q_{\sigma}(x_{t-1}|x_t,x_0) = \mathcal{N}(\sqrt{\alpha_{t-1}}x_0 + \sqrt{1-\alpha_{t-1}-\sigma_t^2} \cdot \cfrac{x_t - \sqrt{\alpha_t}x_0}{\sqrt{1-\alpha_t}}, \sigma_t^2I)
$$

- DDIM의 reverse process

![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_2.png)

여기서  $\sigma_t = \eta\sqrt{(1-\alpha_{t-1}) / (1-\alpha_t)} \sqrt{1-\alpha_t/\alpha_{t-1}}$이다. $\eta$=1인 경우 DDPM이 되고 stochastic해지며,  $\eta$=0인 경우 DDIM이 된다. 

본 논문에서는 DDIM의 reverse process 식을 아래의 식으로 간략하게 쓴다.  "predicted $x_0$"을 $\mathrm{P}_t(\epsilon_t^{\theta}(x_t))$ 라고 표현하고, "direction pointing to $x_t$"부분을 $\mathrm{D}_t(\epsilon_t^{\theta}(x_t))$라고 표현한다.  

$$
x_{t-1} = \sqrt{\alpha_{t-1}}\mathrm{P}_t(\epsilon_t^{\theta}(x_t)) + \mathrm{D}_t(\epsilon_t^{\theta}(x_t)) + \sigma_t\mathcal{z_t}
$$

또한, 간결성을 위해 $\mathrm{P}_t(\epsilon_t^{\theta}(x_t))$ 는 $P_t$로  $\mathrm{D}_t(\epsilon_t^{\theta}(x_t))$는 $D_t$로 표현하고,  $\eta\ne0$일 때를 제외하고는  $\sigma_t\mathcal{z_t}$를 생략한다. 

## **Image Manipulation with CLIP**

CLIP은 Image Encoder $E_I$와 Text Encoder $E_T$로 멀티모달 임베딩을 학습하며, 유사성은 이미지와 텍스트 간의 유사성을 나타낸다. Editied image와 target distription 사이의 코사인 거리를 이용한 directional loss를 이용하여 mode collapse없이 균일한 editing을 하였다. 

$$
\mathcal{L}_{direction} (x^{edit}, y^{target};x^{source},y^{source}) := 1 - \cfrac{\Delta I \cdot \Delta T}{\parallel\Delta I\parallel \parallel\Delta T\parallel}
$$

$\Delta T = \mathrm{E}_T(y^{target}) - \mathrm{E}_T(y^{source})$  

$\Delta I = \mathrm{E}_I(x^{edit}) - \mathrm{E}_I(x^{source})$

- $x^{edit}$: edited image
- $y^{target}$: target description
- $x^{source}$: original image
- $y^{source}$: source description

## 3. **Discovering Semantic Latent Space In Diffusion Models**

해당 파트에서는 왜 기존에 방법들이 reverse process를 제어를 잘 하지 못했는지에 대해 설명하고, 생성 과정을 제어할 수 있는 기술에 대해 소개한다. 

## 3.1. Problem

Semantic latent manipulation을 하는 첫 번째 방법은 2에서 설명한 것처럼 텍스트 프롬프트가 주어졌을 때 CLIP loss를 최적화 하도록  x_T를 업데이트 하여 x_0를 조정하는 것이다. 하지만 이 방법은 이미지가 왜곡되거나 잘못된 조작으로 이어지게 된다. 

다른 접근 방식으로는 각 샘플링 단계에서 네트워크가 예측한 노이즈 $\epsilon_t^{\theta}$를 원하는 방향으로 이동시키는 것이다. 하지만 이 방법은 $P_t$와 $D_t$의 중간 변화가 상쇄되어 기존 latent variable과 다르지 않게 된다. 

이에 대한 증명은 본 논문의 Appendix C에 수록되어 있다.

- 증명 (Appendix C)
    
    ![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_3.png)
    
    ![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_4.png)
    

## **3.2 Asymmetric Reverse Process(Asyrp)**

위에서 설명한 문제를 해결하기 위해 본 논문에서는 Asyrp를 제안한다. 기존 방식이 $P_t$와 $D_t$의 중간 변화가 상쇄되어 원하는 효과를 내지 못했는데, 이를 해결하기 위해 $P_t$와 $D_t$를 비대칭적으로 동작하게 하는 것이다. $x_0$를 예측하는 $\mathrm{P}_t$에서는 shifted epsilon $\tilde{\epsilon}_t^{\theta}(x_t)$을 사용하고, latent variable로 돌아가는 $\mathrm{D}_t$에서는 non-shifted epsilon $\epsilon_t^{\theta}$을 사용한다. Asyrp를 식으로 표현하면 다음과 같다.

$$
x_{t-1} = \sqrt{\alpha_{t-1}}\mathrm{P}_t(\tilde{\epsilon}_t^{\theta}(x_t)) + \mathrm{D}_t(\epsilon_t^{\theta}(x_t))
$$

Loss는 2에서 소개한 $\mathcal{L}_{direction}$을 사용하여 재구성하였다. Edit을 하지 않은 $\mathrm{P}_t^{source}$와 edit한 $\mathrm{P}_t^{edit}$을 사용한다. Loss식은 다음과 같다. 

$$
\mathcal{L}^{(t)} = \lambda_{CLIP}(\mathrm{P}_t^{edit}, y^{ref};\mathrm{P}_t^{source},y^{source}) + \lambda_{recon}|\mathrm{P}_t^{edit} - \mathrm{P}_t^{source}|
$$

전체적인 reverse process를 그림으로 나타내면 아래와 같다. 

![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_5.png)

x_t로 direction할 때는 원래 DDIM의 노이즈를 사용하고, x_0을 predict할 때는 shifted epsilon을 사용한

## 3.3 h-space

{% include figure image_path="/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_6.png" caption="U-Net structure and h-space" %}

U-net 구조에서 인코더의 가장 깊은 feature map인 $h_t$ (노란색 박스)를 선택하여 $\epsilon_t^{\theta}$를 제어한다. $h_t$는 spatial resolution이 작고 높은 수준의 semantics를 가지고 있다. 

$h_t$를 이용한 샘플링 방정식은 아래와 같이 된다. 

![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_7.png)

위 식에서  $\epsilon_t^{\theta}(x_t|\Delta{h_t})$는 original featuremap $h_t$에 $\Delta{h_t}$를 추가한다.

h-space는 다음과 같은 속성을 가지고 있다. 

- 동일한 $\Delta{h_t}$는 다른 샘플에 동일한 효과를 가져온다.
- 선형 스케일링 $\Delta{h_t}$는 음수 스케일에서도 속성 변화의 크기를 제어한다.
- 여러 개의 $\Delta{h_t}$를 추가하면 해당되는 여러 속성을 동시에 조작한다.
- $\Delta{h_t}$는 화질 저하 없이 결과 이미지의 품질을 보존한다.
- $\Delta{h_t}$는 다른 시간 가격 t에 걸쳐 대게 일관성이 있다.

## 3.4 **Implicit Neural Directions**

여러 시간 각격에 대해 $\Delta{h_t}$를 직접 최적화 하려면 학습에 많은 iteration이 필요하고, learning rate와 scheduling을 선택해야 하는 문제점이 있다. 이를 해결하기 위해 $h_t$와 $t$가 주어졌을 때 $\Delta{h_t}$를 만들어내는 implicit function $f_t(h_t)$를 정의한다. 이것은 timestep t로 연결 된 2개의 1x1 convolution으로 구현하였다. 

![스크린샷 2023-12-01 오전 1.49.46.png](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/%25e1%2584%2589%25e1%2585%25b3%25e1%2584%258f%25e1%2585%25b3%25e1%2584%2585%25e1%2585%25b5%25e1%2586%25ab%25e1%2584%2589%25e1%2585%25a3%25e1%2586%25ba_2023-12-01_%25e1%2584%258b%25e1%2585%25a9%25e1%2584%258c%25e1%2585%25a5%25e1%2586%25ab_1.49.46.png)

# 4. Generative Process Design

이 파트에서는 전체적인 editing process에 대해 설명한다. 전체적인 process는 세 단계로 이루어진다. 

![스크린샷 2023-12-01 오전 1.54.08.png](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/%25e1%2584%2589%25e1%2585%25b3%25e1%2584%258f%25e1%2585%25b3%25e1%2584%2585%25e1%2585%25b5%25e1%2586%25ab%25e1%2584%2589%25e1%2585%25a3%25e1%2586%25ba_2023-12-01_%25e1%2584%258b%25e1%2585%25a9%25e1%2584%258c%25e1%2585%25a5%25e1%2586%25ab_1.54.08.png)

1. Asyrp을 이용한 editing
2. 기존 denoising 
3. Quality boosting

본 논문에서는 각 단계의 길이를 정량화 할 수 있는 공식을 설계했다. 

## 4.1. Editing process with Asyrp

생성 과정을 수정하여 semantic을 바꾸는 초기 단계이다. 아래의 식으로 구간 [T,t]에서의 editing strength를 정의한다. 

![스크린샷 2023-12-01 오전 1.59.50.png](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/%25e1%2584%2589%25e1%2585%25b3%25e1%2584%258f%25e1%2585%25b3%25e1%2584%2585%25e1%2585%25b5%25e1%2586%25ab%25e1%2584%2589%25e1%2585%25a3%25e1%2586%25ba_2023-12-01_%25e1%2584%258b%25e1%2585%25a9%25e1%2584%258c%25e1%2585%25a5%25e1%2586%25ab_1.59.50.png)

편집 간격이 짧을수록 $\xi_t$가 낮아지고, 편집 간격이 길수록 결과 이미지에 더 많은 변화가 생긴다. 충분한 변화를 줄 수 있는 한에서 가장 최소의 Editing interval을 찾는 것이 $t_{edit}$을 결정하는 최고의 방법이다. 저자들은 실험적인 결과를 통해 $\mathrm{LPIPS}(x, \mathrm{P}_t)$ *= 0.33*인 t시점을 *$t_{edit}$*으로 결정하였다.

 저자들은 실험을 통해 $\text{LIPPS}(x,P_{t_{edit}})=0.33$인 t시점을 $*t_{edit}*$으로 결정하였다. 이 지점이 충분한 변화를 줄 수 있으면서 가장 최소의 editing interval이었다. 

아래의 그림은 다양한 $\mathrm{LPIPS}(x, \mathrm{P}{t_{edit}})$에 따른 생성 결과를 나타낸 그림이다.

![다양한 $\mathrm{LPIPS}(x, \mathrm{P}{t_{edit}})$에 따른 생성 결과](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_8.png)



## 4.2. **Quality Boosting With Stochastic Noise Injection**

DDIM은 stochasticity를 제거하여 거의 완벽한 inversion을 달성하지만, stochasticity이 이미지 품질을 향상 시킨다는 결과가 있다. 따라서, 본 논문에서 boosting interval에서는 이미지 품질을 향상 시키기 위해서 이 간격에서는 stochastic noise를 주입한다. 

부스팅 간격이 길수록 품질이 높아지지만, 지나치게 긴 간격일 때에는 콘텐츠가 변형 될 수 있다. 따라서 저자들은 콘텐츠의 변화를 최소화 하면서 충분한 화질 부스팅을 제공하는 최단 간격을 찾고자 하였다. 본 논문에서는 이미지 노이즈를 quality boosting의 capacity로 간주하고, 원본 이미지와 비교하여 $x_t$의 노이즈 양을 나타내는 quality deficiency를 아래와 같이 정의했다. 

$$
\gamma_t = \mathrm{LPIPS}(x, x_t)
$$

저자들은 실험을 통해 $\gamma_t$ = 1.2인 t시점을 $t_{boost}$로 설정하였다.

아래의 그림은 quality boosting을 적용할 때와 적용하지 않았을 때의 결과 차이를 나타낸 그림이다.

![quality boosting을 적용할 때와 적용하지 않았을 때의 결과 차이](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_9.png)



## 4.3 Overall Process of Image Editing

$t_{edit}$과 $t_{boost}$를 이용한 전체적인 generative process는 아래와 같이 정리된다. 

![스크린샷 2023-12-01 오전 2.16.53.png](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/%25e1%2584%2589%25e1%2585%25b3%25e1%2584%258f%25e1%2585%25b3%25e1%2584%2585%25e1%2585%25b5%25e1%2586%25ab%25e1%2584%2589%25e1%2585%25a3%25e1%2586%25ba_2023-12-01_%25e1%2584%258b%25e1%2585%25a9%25e1%2584%258c%25e1%2585%25a5%25e1%2586%25ab_2.16.53.png)

# 5. Experiments

- 데이터셋과 모델
    - CelebA-HQ, SUN-bedroom/-church 데이터셋을 이용하여 DDPM++를 학습
    - FHQ-dog 데이터셋을 이용하여 iDDPM을 학습
    - METFACES 데이터셋에서 ADM with P2-weighting를 사용해 학습
    
    → 모든 모델들은 pretrained checkpoint를 활용했으며 frozen상태를 유지시켰다. 
    

## **5.1 Versatility of h-space with Asyrp**

다양한 데이터셋에 대한 Asyrp에 editing result이다. 

![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_10.png)

위의 그림을 볼 수 있듯, 다양한 attribute들의 특성을 잘 반영해서 이미지를 조정한 것을 알 수 있다. 심지어 훈련에 포함되지 않은 속성인 {department, factory, temple}에 대해서도 합성할 수 있었다. 무엇보다 모델을 fine tuning하지 않고 inference 중에서 Asyrph를 사용하여 h-space의 bottle neck feature maps만 이동 시킨 결과라는 것이 놀랍다.

## **5.2 Quantitive Comparison**

Fine-tuning없이 다양한 diffusion 모델과 결합 할 수 있는 점을 고려할 때, 비슷한 경쟁자를 찾을 수 없었다. 따라서 전체 모델을 fine-tuning하여 이미지를 편집하는 DiffusionCLIP과 비교를 하였다. 80명의 참가자에게 총 40개의 원본 이미지 세트와 Asyrp의 결과와 DiffusionCLIP의 결과를 비교하도록 하였다. (품질, 자연스러움, 전반적인 선호도 고려) 그 결과는 아래와 같다. 

![스크린샷 2023-12-01 오전 2.29.32.png](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/%25e1%2584%2589%25e1%2585%25b3%25e1%2584%258f%25e1%2585%25b3%25e1%2584%2585%25e1%2585%25b5%25e1%2586%25ab%25e1%2584%2589%25e1%2585%25a3%25e1%2586%25ba_2023-12-01_%25e1%2584%258b%25e1%2585%25a9%25e1%2584%258c%25e1%2585%25a5%25e1%2586%25ab_2.29.32.png)

모든 관점에서 Asyrp가 DiffusionCLIP을 능가하는 것을 볼 수 있었다. 

![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_11.png)


## **5.3 Analysis on h-space**

### Homogeneity

아래 그림은 $\epsilon$-space와 비교한 h-space의 homogeneity를 보여준다. 하나의 이미지에 대해 $\Delta h_t$를 최적화하면 다른 입력 이미지에도 동일한 속성 변경이 발생한다. 반면에 하나의 이미지에 최적화 된  $\Delta \epsilon_t$는 다른 입력 이미지를 왜곡한다. 

![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_12.png)

### Linearity

아래 그림을 통해 $\Delta h_t$의 선형 스케일링이 시각적 속성의 변화량을 반영하는 것을 확인할 수 있다. 놀라벡도 훈련 중에는 볼 수 없는 음의 스케일로도 일반화가 된다. 

![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_13.png)

또한, 아래 그림처럼 서로 다른 $\Delta h_t$의 조합이 결과 이미지에서 결합된 semantic change를 보여주는 것을 확인할 수 있다. 

![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_14.png)

### **Robustness**

아래 그림은 h-space와 $\epsilon$-space에서 random noise를 주입했을 때의 결과를 비교한 것이다.  h-space는 random noise가 추가되었어도 이미지에 큰 변화가 없으며 많은 noise가 추가되었을 경우에도 이미지 왜곡은 거의 없고 semantic change만 발생한다. 반면에 $\epsilon$-space의 경우에는 random noise가 추가된 경우 이미지 왜곡이 심하게 발생한다.

![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_15.png)

### **Consistency across time steps**

모든 샘플의 $\Delta h_t$는 균일하며, 평균 $\Delta h^{mean}$으로 대체해도 비슷한 결과가 나온다. 최상의 품질과 조작을 위해 $\Delta h_t$를 사용하지만, 간결성을 위해 $\Delta h^{mean}$ , 또는 약간의 절충인 $\Delta h^{global}$을 사용할 수 있다. 이 때에도 $\Delta h_t$를 사용했을 때와 비슷한 결과를 얻을 수 있다.

![Untitled](/assets/images/2023-12-01-diffusion_models_already_have_a_semantic_latent_space/untitled_16.png)

# 6. **Conclusion**

본 논문에서는 사전 훈련된 diffusion model을 위해 latent semantic space h-space에서 이미지 편집을 용이하게 하는 새로운 생성 프로세스인 Asyrp을 제안하였다. h-space는 GAN의 latent space와 마찬가지로 homogeneity, Linearity, Robustness, Consistency across timesteps 등의 좋은 특성을 가지고 있다. 전체 editing process는 시간 단계별로 editing strength와 quality deficiency를 측정하여 다양한 편집과 높은 품질을 달성하도록 설계되었다. 
