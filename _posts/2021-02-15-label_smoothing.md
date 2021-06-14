---
title: "[Tips] 라벨 스무딩을 이용한 모델 성능 개선"
subtitle: 라벨 스무딩을 이용한 모델 성능 개선
categories: Tips
date: 2021-02-15 15:06:35 +0900
tags:
  - 라벨스무딩
  - 딥러닝 모델 성능 개선
  - label smoothing
toc: true
toc_sticky: true
---

# 1. Label smoothing이란?

Hard target을 soft target으로 바꾸는 것으로 라벨 스무딩을 이용하면 모델의 일반화 성능이 향상된다고 알려져 있습니다.
간단히 말하자면, 아래의 식으로 hard target을 soft target으로 바꾸어 모델의 over confidence 문제를 해결할 수 있기에 모델의 일반화 성능이 향상됩니다. 라벨 스무딩이 잘 동작하는 이유에 대한 자세한 설명이 궁금하시다면 해당 [링크](https://ratsgo.github.io/insight-notes/docs/interpretable/smoothing)에서 확인할 수 있습니다.

$$\tag{1} y(k)' = y(k)(1-\alpha)+\alpha/K\\\scriptsize{\text{($k$: 현재 클래스의 index, $y(k)$: ground truth, $K$: 총 클래스 개수, $\alpha$: 최적화 해야 할 하이퍼파라미터)}} $$


예: $\alpha$가 0.1일 때 
<br>Hard target: [0,1,0,0]
<br>Soft target:  [0.025, 0.925,0.025,0.025]

# 2. Cross entropy에서의 적용

라벨스무딩을 cross entropy에 적용하는 방법은 단순합니다. 그저 일반적인 cross entropy식에서 truth label인 $y(k)$를 (1)의 식으로 구한 $y'(k)$로 바꾸면 됩니다. 

Label smoothing을 적용한 truty 라벨에 대해 cross entropy를 적용하면 다음과 같이 됩니다. $p(k)$와 $y(k)$라는 두 확률 분포가 있을 때 두 확률 분포 사이의 cross entrophy는 아래와 같습니다. 여기서 $p(k)$는 모델의 예측 값이고, $y(k)$는 ground truth라고 가정합니다. 

$$\tag{2} H(y,p) =-\sum_{k=1}^K\log (p(k))y(k)\\\scriptsize{\text{($p(k)$: predicted, $y(k)$: ground truth)}}$$

$H(y',p)$을 구하기 위해 위의 식에서 $y(k)$ 자리에 라벨 스무딩을 적용한 $y(k)'$를 대입합니다. 이 때, $y(k)'$를 계산하는 식 $(1)$의 마지막 항 $\alpha/K$는  uniform distribution에 $\alpha$가 곱해진 것으로 여길 수 있습니다. 따라서 $y(k)'$는 다음과 같이 쓸 수 있습니다. 여기서 $u(k)$는 uniform distribution을 의미합니다.

$$\tag {3} y(k)' = y(k)(1-\alpha)+\alpha u(k)$$

위 식 $(3)$을 대입하여 $H(y',p)$를 구하면 아래의 식으로 표현할 수 있습니다.

$$\tag{4} H(y',p) = -\sum_{k=1}^K\log (p(k))\{(1-\alpha)y(k)+\alpha(u(k))\}\\ = (1-\alpha)H(q,p)+\alpha H(u,p)$$

# 3. Tensorflow

Tensorflow에서는 [BinaryCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy)와 [CategoricalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy)함수에 이미 구현되어 있습니다.  간단하게 `label_smoothing`에 값을 넣으면 됩니다. 

```python
tf.keras.losses.BinaryCrossentropy(
    from_logits=False, label_smoothing=0, reduction=losses_utils.ReductionV2.AUTO,
    name='binary_crossentropy'
)
```

# 4. Pytorch

OpenNMT [코드 예시](https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186)
```python
class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')
```

[코드 예시 2](https://programmersought.com/article/27102847986/)
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
```

아래 코드를 보면 먼저 `.log_softmax` 함수를 통해 log softmax를 구합니다. 이것은 나중에 cross entropy loss를 계산하기 위함입니다. Log softmax와 target을 곱한 것의 음수를 취한 것이 cross entrophy loss가 됩니다. `true_dist.fill_(self.smoothing / (self.cls - 1))` 을 통해 $(1)$의 뒷항인 $\alpha/K$을 구합니다.  `scatter_` 함수를 통해 target의 index에 해당하는 위치에 $(1-\alpha)$을 넣게 됩니다. 그렇게 만들어진 새로운 target의 `true_dist` 와 log softmax값이 들어간 `pred` 를 곱해 최종적인 cross entropy loss를 만듭니다. 

위의 코드가 이해가 잘 안된다면 먼저 [이 글](http://www.gisdeveloper.co.kr/?p=8668)을 읽어보길 추천합니다. CrossEntropyLoss를 구현하는 여러 버전의 코드가 있습니다. 

# 5. Reference

- [https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06](https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06)
- [https://medium.com/towards-artificial-intelligence/how-to-use-label-smoothing-for-regularization-aa349f7f1dbb](https://medium.com/towards-artificial-intelligence/how-to-use-label-smoothing-for-regularization-aa349f7f1dbb)
- [https://ratsgo.github.io/insight-notes/docs/interpretable/smoothing](https://ratsgo.github.io/insight-notes/docs/interpretable/smoothing)
- [C. Szegedy, et al, "Rethinking the Inception Architecture for Computer Vision," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 2818-2826, doi: 10.1109/CVPR.2016.308.](https://arxiv.org/abs/1512.00567)
- [http://www.gisdeveloper.co.kr/?p=8668](http://www.gisdeveloper.co.kr/?p=8668)
