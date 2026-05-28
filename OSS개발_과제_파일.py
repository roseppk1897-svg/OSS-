{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyNLGvkb/bMSG0IbE9aISm2P",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/roseppk1897-svg/OSS-/blob/OSS-test/OSS%EA%B0%9C%EB%B0%9C_%EA%B3%BC%EC%A0%9C_%ED%8C%8C%EC%9D%BC.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QVBMN2kbf5b5",
        "outputId": "2b3b6fbc-192c-4514-949c-3221f53e88a4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "=================================================================\n",
            "---------- 데이터 로드 결과 ----------\n",
            "샘플 개수: 569\n",
            "특성 개수: 30\n",
            "=================================================================\n",
            "[데이터 분할 결과 요약]\n",
            "학습 데이터 크기: (455, 30)\n",
            "테스트 데이터 크기: (114, 30)\n",
            "=================================================================\n",
            "[실험 1] 기본 파라미터(n_neighbors=6) k-NN 모델 학습 진행 중입니다.삐빅.\n",
            "[실험 1] 기본 모델 테스트 정확도: 0.9649122807017544\n",
            "=================================================================\n",
            "[실험 2] 파라미터 변경(n_neighbors=3) k-NN 모델 학습 진행 중...입니다. 삐빅.\n",
            "[실험 2] 파라미터 변경 모델 테스트 정확도: 0.9298245614035088\n",
            "=================================================================\n",
            "\n",
            "\n",
            "=================================================================\n",
            "[최종 실험 결과 비교 요약]\n",
            "  * 실험 1 정확도 (k=6 일 때) -> 0.9649122807017544\n",
            "  * 실험 2 정확도 (k=3 일 때) -> 0.9298245614035088\n",
            "=================================================================\n",
            "\n",
            "\n",
            "=================================================================\n",
            "  * 두 실험의 정확도 차이 결과 -> -0.03508771929824561\n",
            "==================== 실험 최종 종료 =============================\n"
          ]
        }
      ],
      "source": [
        "\n",
        "#유방암 데이터셋 분류\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn import datasets, metrics\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "\n",
        "# 1. 데이터 가져오기\n",
        "cancer = datasets.load_breast_cancer()\n",
        "X = cancer.data\n",
        "y = cancer.target\n",
        "\n",
        "print(\"=\" * 65)\n",
        "print(\"---------- 데이터 로드 결과 ----------\")\n",
        "print(\"샘플 개수:\", len(X))\n",
        "print(\"특성 개수:\", X.shape[1])\n",
        "print(\"=\" * 65)\n",
        "\n",
        "\n",
        "# 2. 데이터 나누기 (학습 8 : 테스트 2)\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=42\n",
        ")\n",
        "\n",
        "print(\"[데이터 분할 결과 요약]\")\n",
        "print(\"학습 데이터 크기:\", X_train.shape)\n",
        "print(\"테스트 데이터 크기:\", X_test.shape)\n",
        "print(\"=\" * 65)\n",
        "\n",
        "\n",
        "# [실험 1] k=6 기본 모델 돌려보기\n",
        "print(\"[실험 1] 기본 파라미터(n_neighbors=6) k-NN 모델 학습 진행 중입니다.삐빅.\")\n",
        "knn1 = KNeighborsClassifier(n_neighbors=6)\n",
        "knn1.fit(X_train, y_train)\n",
        "\n",
        "# 테스트 데이터 예측\n",
        "y_pred1 = knn1.predict(X_test)\n",
        "# 정확도 계산\n",
        "scores1 = metrics.accuracy_score(y_test, y_pred1)\n",
        "print(\"[실험 1] 기본 모델 테스트 정확도:\", scores1)\n",
        "print(\"=\" * 65)\n",
        "\n",
        "# [실험 2] 파라미터 k=3으로 바꿔서 실험하기\n",
        "print(\"[실험 2] 파라미터 변경(n_neighbors=3) k-NN 모델 학습 진행 중...입니다. 삐빅.\")\n",
        "knn2 = KNeighborsClassifier(n_neighbors=3)\n",
        "knn2.fit(X_train, y_train)\n",
        "\n",
        "# 테스트 데이터 예측\n",
        "y_pred2 = knn2.predict(X_test)\n",
        "# 정확도 계산\n",
        "scores2 = metrics.accuracy_score(y_test, y_pred2)\n",
        "print(\"[실험 2] 파라미터 변경 모델 테스트 정확도:\", scores2)\n",
        "print(\"=\" * 65)\n",
        "\n",
        "# 4. 결과 비교창\n",
        "print(\"\\n\")\n",
        "print(\"=\" * 65)\n",
        "print(\"[최종 실험 결과 비교 요약]\")\n",
        "print(\"  * 실험 1 정확도 (k=6 일 때) ->\", scores1)\n",
        "print(\"  * 실험 2 정확도 (k=3 일 때) ->\", scores2)\n",
        "print(\"=\" * 65)\n",
        "print(\"\\n\")\n",
        "\n",
        "#실험1, 2 결과를 통해 정확도 차이 구하기\n",
        "print((\"=\"*65))\n",
        "print(\"  * 두 실험의 정확도 차이 결과 ->\", scores2 - scores1)\n",
        "print(\"=\"*20, \"실험 최종 종료\", \"=\"*29)"
      ]
    }
  ]
}