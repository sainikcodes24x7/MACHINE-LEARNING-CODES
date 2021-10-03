{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "PolynomialRegression.py",
      "provenance": [],
      "authorship_tag": "ABX9TyMubQ+13sCirKLw+CnOLvcO",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sainik-khaddar/MACHINE-LEARNING-CODES/blob/main/PolynomialRegression_python.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ieYnYaChkZUt"
      },
      "source": [
        "#IMPORTING THE DATASET"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "y7D77bHlkc9A"
      },
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "t27c0N2i4GaU"
      },
      "source": [
        "#IMPORTING THE DATASET"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iZs5L56wkmlD"
      },
      "source": [
        "df=pd.read_csv('Position_Salaries.csv')\n",
        "X=df.iloc[:, 1:-1].values\n",
        "y=df.iloc[:,-1].values"
      ],
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "blN2nk9F0vbI",
        "outputId": "3bfeda4b-5c27-416a-9fef-a54c3c7479cd"
      },
      "source": [
        "print(y)"
      ],
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[  45000   50000   60000   80000  110000  150000  200000  300000  500000\n",
            " 1000000]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4RXln5vW09AN",
        "outputId": "62a54a5d-f701-4a8b-9d9c-0114e8f78a35"
      },
      "source": [
        "print(X)"
      ],
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 1]\n",
            " [ 2]\n",
            " [ 3]\n",
            " [ 4]\n",
            " [ 5]\n",
            " [ 6]\n",
            " [ 7]\n",
            " [ 8]\n",
            " [ 9]\n",
            " [10]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Q7LWCwtl4KpL"
      },
      "source": [
        "#Training the linear regression model on the whole dataset"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FRzl6RTd4UUJ",
        "outputId": "e6048a67-1166-4902-a991-ad2cc4debcde"
      },
      "source": [
        "from sklearn.linear_model import LinearRegression\n",
        "lin_reg=LinearRegression()\n",
        "lin_reg.fit(X,y)"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rP65Bi759Hd3"
      },
      "source": [
        "#Training the Polynomial regression model on the whole dataset"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "92o-BvMT9N0g",
        "outputId": "5c78f8e3-61be-4b24-e292-9376063e778c"
      },
      "source": [
        "from sklearn.preprocessing import PolynomialFeatures\n",
        "polyreg=PolynomialFeatures(degree=4)\n",
        "X_poly=polyreg.fit_transform(X)\n",
        "lin_reg2=LinearRegression()\n",
        "lin_reg2.fit(X_poly,y)"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yz-hR6tI25jG"
      },
      "source": [
        "#VISUALIZING THE LINEAR REGRESION RESULTS"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "NxljdIL52-lc",
        "outputId": "8f2eaec2-1384-41ac-ffb6-edc8a4b5ab96"
      },
      "source": [
        "plt.scatter(X,y,color='red')\n",
        "plt.plot(X,lin_reg.predict(X),color='blue')\n",
        "plt.title('Truth or bluff Liunear Regression')\n",
        "plt.xlabel('LEVEL')\n",
        "plt.ylabel('SALARY')\n",
        "plt.show()"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PwuNO0tHocn1"
      },
      "source": [
        "#VISUALIZING THE POLYNOMIAL REGRESION RESULTS"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "-2CdjzwKB2v8",
        "outputId": "74f0425b-118a-4286-8ac9-aaccbb18afe4"
      },
      "source": [
        "plt.scatter(X,y,color='red')\n",
        "plt.plot(X,lin_reg2.predict(X_poly),color='blue')\n",
        "plt.title('Truth or bluff Ploynomial Regression')\n",
        "plt.xlabel('LEVEL')\n",
        "plt.ylabel('SALARY')\n",
        "plt.show()"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYIAAAEWCAYAAABrDZDcAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3deZgU1dn38e8NgwKCoMCjsjkoqPAocSEwaiIoJqIRiUYRRBPiQpJHE03ibmJMFE2MbzaXKMYtzAQkuGGiQeOCJhFwcAEFFwSBAQwDIqKALHO/f5wa6WlmZbqmuqd/n+vqa7qrTlfdXd1z7qpTdU6ZuyMiIvmrRdIBiIhIspQIRETynBKBiEieUyIQEclzSgQiInlOiUBEJM8pEeQxM3vfzI5vonUNMbOyWubfb2Y31HNZbczscTNbZ2Z/jabdYGarzeyDBsZVaGZuZgUNeV+SzOxJM/tWPcs22Xfc1MzsEzPbL+k4mgMlgiwW/dArHxVmtjHl9ZgGLqveFW0OOB3YC+jk7meYWU/gx0A/d987vXCUhCqi7bbezN42s283ddCZ4u4nuvsDjV1O9JvYHG2XD83saTM7KBMxNgV3b+fui5KOozlQIshi0Q+9nbu3A5YCw1OmlVSWy7a92SaIZ1/gHXffGr3uCaxx91W1vGdFtB13B64A7jazfjHHmQtujrZLN2A5cE+mV5Btv0/ZkRJBDqpsZjGzK6KmkPvMbKyZ/SutnJtZbzMbB4wBLo/2/h5PKXaomc2NmlkeNLPWNayzhZn9xMyWmNkqM/uzmXWI5lU2r5xnZkuBZ2uJ/eqoCef9mo5q6vgsPweuBc6MPst3gKeBrtHr+2vbdh48CqwFdkgEZtbVzKZFe8gLzeyCaPreZrbBzDqllD3czMrNrFVlzGZ2i5mtNbPFZnZiXcuN5l1nZn81s+LoiGWemR1gZldF23qZmX01pfzzZnZ+9Hx/M3vWzNZE27XEzDrWtg1q2C4bgSnAoWkxPxR9xsVm9oOUeW3M7IHosy4ws8stpekv+n6vMLO5wKdmVmBmRWb2HzP7yMxeN7MhKeXHmtmi6PMvrvxtRN/5jOj3udrMHkx5j5tZ7+h5h+g3WR79Rn9iZi1Sll3jdyNKBLlsb2BPwt7xuNoKuvsEoIRo78/dh6fMHgkMA3oB/YGxNSxmbPQ4FtgPaAfcllZmMNAXOKGWmDsT9j6/BUwwswNri72az/Iz4Ebgweiz3AWcSLTH7+41xQ98ntBOBToC86opMhkoA7oSmqBuNLPj3P0D4HnC9qp0DjDZ3bdErwcBb0ef8WbgHjOz2pabsqzhwERgD+BVYDrh/7Mb8Avgrpo+EnBTtNy+QA/gutq2QbULMdsNGA0sjF63AB4HXo9iGApcYmaV3+3PgELCb+ErwNnVLHY08DXCtt4L+DtwA+F3eynwkJl1idb9B+BEd28PHAW8Fi3jeuApwnbpDtxaw0e4FegQxTMY+CaQ2vxX23cj7p5zD+BeYBXwRj3LjwTmA28Cf0k6/p38zO8Dx0fPhwCbgdYp88cC/0p7jwO9o+f3AzdUs8yzU17fDNxZw/qfAf4v5fWBwBaggFAhOLBfLfEPAbYCu6VMmwL8ND2+enyW64DitGWX1bHuCuAj4ENCJTMqmlcZewGhEt0GtE95703A/dHzM4F/R89bAh8AA1NiXpjyvrbRcveux3KvA55OmTcc+ARoGb1uHy2rY/T6eeD8Gj7r14FXq/vdVFP2fmBTtF0qgMVA/2jeIGBpWvmrgPui54uAE1LmnZ/6HUTrPTfl9RXAxLTlTSfsEOwWxfANoE1amT8DE4Du1cTvQO/ou9hMOEdUOe87wPN1fTdJ/19nyyNXjwjuJ+zF1snM+hB+wEe7+/8Cl8QYV1Mqd/dNGVhO6lU2Gwh7+tXpCixJeb2EUHnulTJtWR3rWuvun6Yto2s942ysFe7e0d33dPdD3X1yNWW6Ah+6+/qUaUsIe8QAjwH9zKwXYS94nbvPTin7+bZ09w3R03b1WC7Af1OebwRWu/u2lNeVy6rCzPYys8lmttzMPgaKCXu99XWLu3ckJMSNhAQP4Uiza9SM85GZfQRczfbvuytVv+/qvvvUafsCZ6Qt70vAPtFv4kzgu8BKM/u7bT9pfTnhqGe2mb1pZudWs57OQCt2/H2mbt+avhshR5uG3P0Fwp7d56K20n+Y2RwzezHlh3QBcLu7r43eW9sJxVySPmzsp4Q9HSC0addRvqFWEP6ZK/Uk7OGnVmB1rWOPqBkgdRkrqilX12eJywpgTzNrnzKtJ+EkKlHinUJoBjmH0JTT6OU20o2E7X6Iu+8exdbgJg93XwpcDPzezNoQKvHFUfKsfLR395Oit6wkNNVU6lHdYlOeLyMcEaQubzd3/2W0/unu/hVgH+At4O5o+gfufoG7dyXs5d9ReV4gxWrC0Wn67zMT2zcv5GQiqMEE4PvufgSh/fGOaPoBwAFm9m8zm2lm9TqSyEGvA/9rZodaOOF7Xdr8/xLaT3fWJOCHZtbLzNqxvZ1+ax3vS/dzM9vFzL4MnAz8tZoydX2WWLj7MuA/wE1m1trM+gPnEfayK/2Z0NRwCvVMBPVc7s5qT2hGWmdm3YDLdnZB7v40IWmNA2YD66MTvm3MrKWZHWxmX4yKTwGuMrM9ovVeVMfii4HhZnZCtKzWFi566B4d1YyIdhI+iz5PBYCZnWFmlQlnLSG5VKTFvS2KZ7yZtTezfYEfkZntmxeaRSKIKqajgL+a2WuEE2v7RLMLgD6EduLRhMsGG3xVRbZz93cIJxX/CbwL/CutyD2EZo2PzOzRnVjFvYSK7wVCW/Im4PsNXMYHhH/mFYST199197fSC9Xjs8RpNKGZZAXwCPAzd/9nSmz/JlREr7j7kmqXsBPLbYSfA4cD6wgnYx9u5PJ+TWiOKSAk6kMJ3/dq4E+EE7IQvp+yaN4/gamESrxaUTIcQWheKiccIVxGqINaECruFYQj/cHA96K3fhGYZWafANOAi736vgPfJxxJLiL8Xv5C+M1KPVh08iTnmFkh8Dd3P9jMdgfedvd9qil3JzDL3e+LXj8DXOnuLzdlvNJ8mNmzhIsO/pR0LNnCzL5HOAE/OOlYpOGaxRGBu38MLDazMwAs+EI0+1HC0QBm1pnQVKTeiLJToqaRw4EH6yrbnJnZPmZ2dHQ57oGEnt2PJB2X7JycTARmNgl4CTjQQseq8wgdps4zs9cJl4mOiIpPB9aY2XzgOeAyd1+TRNyS28zsAUIzyCVpVwDlo10ITbDrCR0IH2P7eTnJMTnbNCQiIpmRk0cEIiKSOTk3GFTnzp29sLAw6TBERHLKnDlzVrt7l+rm5VwiKCwspLS0NOkwRERyipnVeLmzmoZERPKcEoGISJ5TIhARyXNKBCIieU6JQEQkz8WWCMzsXgu32XujhvlmZn+wcNu+uWZ2eFyxiIjktJISKCyEFi3C35KSut7RIHEeEdxP7TePOZEwKmgfwrC3f4wxFhGR3FRSAuPGwZIl4B7+jhuX0WQQWyKo7uYxaUYAf/ZgJtDRzHYYPVREJK9dcw1s2FB12oYNYXqGJHmOoBtVb2VXRtVby33OzMaZWamZlZaXlzdJcCIiWWHp0oZN3wk5cbLY3Se4+wB3H9ClS7U9pEVEmqeePQF4kmFsoM0O0zMhyUSwnKr3Oe2O7jEqIlLV+PEsbH0wJ/Ekd/B/YVrbtjB+fMZWkWQimAZ8M7p6qAhY5+4rE4xHRCT7jBlD8YklGBWMZjLsuy9MmABjxmRsFbENOhfdPGYI0NnMyoCfAa0A3P1O4AngJGAhsAH4dlyxiIjkKncontuf44ZCt3+WxbKO2BKBu4+uY74DF8a1fhGR5mDWLHjvPfjJT+JbR06cLBYRyVfFxdC6NZx2WnzrUCIQEclSW7bA5MkwYgTsvnt861EiEBHJUtOnw5o1cPbZ8a5HiUBEJEsVF0PnznDCCfGuR4lARCQLrVsHjz0Go0ZBq1bxrkuJQEQkCz38MGzaFH+zECgRiIhkpeJi6N0bBg6Mf11KBCIiWaasDJ57LhwNmMW/PiUCEZEsM2lS6FGcwVEkaqVEICKSZYqLoagoNA01BSUCEZEsMndueJxzTtOtU4lARCSLFBdDQQGMHNl061QiEBHJEtu2wV/+AieeGDqSNRUlAhGRLDFjBixf3jR9B1IpEYiIZIniYmjfHoYPb9r1KhGIiGSBjRth6lQ4/XRo06bu8pmkRCAikgUefxzWr2/6ZiFQIhARyQrFxdC9OwwZ0vTrViIQEUlYeTk8+SScdRa0SKBWViIQEUnYlCmwdWsyzUKgRCAikrjiYujfHw45JJn1KxGIiCRo4UKYOTO5owFQIhARSVRJSRhqevTo5GJQIhARSYh7aBY69thwxVBSlAhERBIye3ZoGmrKkUaro0QgIpKQiROhdWs47bRk41AiEBFJwJYtMHkyjBgBu++ebCxKBCIiCZg+HdasSfZqoUpKBCIiCSguhk6d4IQTko5EiUBEpMl9/DE89hiMGgWtWiUdjRKBiEiTe/hh2LQpO5qFQIlARKTJFRdD794waFDSkQSxJgIzG2Zmb5vZQjO7spr5Pc3sOTN71czmmtlJccYjIpK0sjJ49tlwNGCWdDRBbInAzFoCtwMnAv2A0WbWL63YT4Ap7n4YMAq4I654RESywaRJoUfxmDFJR7JdnEcEA4GF7r7I3TcDk4ERaWUcqLyCtgOwIsZ4REQSV1wMRUWhaShbxJkIugHLUl6XRdNSXQecbWZlwBPA96tbkJmNM7NSMystLy+PI1YRkdjNnRse2XKSuFLSJ4tHA/e7e3fgJGCime0Qk7tPcPcB7j6gS5cuTR6kiEgmlJRAQQGMHJl0JFXFmQiWAz1SXnePpqU6D5gC4O4vAa2BzjHGJCKSiIqKkAiGDYNs25+NMxG8DPQxs15mtgvhZPC0tDJLgaEAZtaXkAjU9iMizc6MGbB8efIjjVYntkTg7luBi4DpwALC1UFvmtkvzOyUqNiPgQvM7HVgEjDW3T2umEREkjJxIrRvD8OHJx3JjgriXLi7P0E4CZw67dqU5/OBo+OMQUQkaRs3wtSpcPrp0KZN0tHsKOmTxSIizd7jj8P69dl3tVAlJQIRkZgVF0O3bjB4cNKRVE+JQEQkRqtXw5NPwllnQcuWSUdTPSUCEZEYTZkCW7dm59VClZQIRERiVFwM/fvDIYckHUnNlAhERGKycCG89FL2niSupEQgIhKTkpIw1PTo0UlHUjslAhGRGLiHZqFjj4Xu3ZOOpnZKBCIiMZg9OzQNZXuzECgRiIjEorgYWreGb3wj6UjqpkQgIpJhW7bA5MkwYgTsvnvd5ZOmRCAikmFPPRU6kuVCsxAoEYiIZNzEidCpE5xwQtKR1I8SgYhIBn38MTz2GIwaBa1aJR1N/SgRiIhk0MMPw6ZNudMsBEoEIiIZVVwM++8PgwYlHUn9KRGIiGTI8uXw7LPhaMAs6WjqT4lARCRDJk0KPYpzqVkIlAhERDKmuBiKiqB376QjaRglAhGRDJg3D15/PfeOBkCJQEQkI4qLoaAARo5MOpKGUyIQEWmkioow5PSwYdClS9LRNJwSgYhII82YEa4YysVmIVAiEBFptOJiaN8eTjkl6Uh2jhKBiEgjbNwIU6fC6adDmzZJR7NzlAhERBrhb38L4wvlarMQKBGIiDTKxInQrRsMHpx0JDtPiUBEZCetXg1PPglnnQUtWyYdzc5TIhAR2UlTpsDWrbndLARKBCIiO624GA45BPr3TzqSxlEiEBHZCe+9By+9lPtHA6BEICKyU0pKwlDTZ52VdCSNF2siMLNhZva2mS00sytrKDPSzOab2Ztm9pc44xERyQT30Cx07LHQvXvS0TReQVwLNrOWwO3AV4Ay4GUzm+bu81PK9AGuAo5297Vm9j9xxSMikimzZ8O778JVVyUdSWbEeUQwEFjo7ovcfTMwGRiRVuYC4HZ3Xwvg7qtijEdEJCOKi6F1azjttKQjyYxaE0G0V7+zugHLUl6XRdNSHQAcYGb/NrOZZjashjjGmVmpmZWWl5c3IiQRkcbZsgUmTw7jCnXokHQ0mVHXEcEcMzsyxvUXAH2AIcBo4G4z65heyN0nuPsAdx/QJRfHeBWRZuOpp0JHsuZwtVCluhLBd4Dfm9ndZrZHA5e9HOiR8rp7NC1VGTDN3be4+2LgHUJiEBHJSsXF0KlTuPdAc1HryWJ3n2Vmg4DvAqVm9iRQkTL/B7W8/WWgj5n1IiSAUUD6hVaPEo4E7jOzzoSmokUN/hQiIk3g44/h0UfhvPOgVauko8mc+lw1tCfwRaAcmENKIqiNu281s4uA6UBL4F53f9PMfgGUuvu0aN5XzWw+sA24zN3X7MTnEBGJ3SOPwKZNzatZCMDcveaZZt8FLgN+DdzltRVuIgMGDPDS0tKkwxCRPHT88fD+++HSUbOko2kYM5vj7gOqm1fXOYIvAUe6+53pScDMjs5UgCIi2W75cnj22XA0kGtJoC51JYKxwFAzu9TMDgYws5PN7D/AbXEHJyKSLSZNCj2Kx4xJOpLMq+scwZ8IV/7MBv5gZiuAAcCV7v5o3MGJiGSL4mIYNAj6NMPrGutKBF8EDnH3CjNrDXwA7K8TuiKST+bNg9dfh9uaaTtIXU1Dn7l7BYC7bwIWKQmISL4pKYGCAhg5MulI4lHXEcFBZjY3em7A/tFrA9zdc/x2DCIitauoCIlg2DBorgMb1JUI+jZJFCIiWWrGDCgrg1tuSTqS+NTVs3hJddPN7EuEHsEXxhGUiEi2KC6G9u1h+PCkI4lPve9HYGaHEYaIOANYDDwcV1AiItng3XfDZaOjRkHbtklHE59aE4GZHUDY8x8NrAYeJPRGPrYJYhMRSczWrXDOOeG+AzfckHQ08arriOAt4EXgZHdfCGBmP4w9KhGRhN10E8yaBQ8+CF27Jh1NvOq6fPQ0YCXwXDQU9VDCFUMiIs1WaSn8/Ocw5qjFjLy8EFq0gMLCcPlQM1RrInD3R919FHAQ8BxwCfA/ZvZHM/tqUwQoItKUNmwITUL7dPiU2149GpYsCWNLLFkC48Y1y2RQr3sWu/un7v4Xdx9OuMHMq8AVsUYmIpKAK6+Et96C+wsuoOPGlVVnbtgA11yTTGAxavDN6919rbtPIJw8FhFpNp5+Gm69FS65BIaWT66+0NKlTRtUE2hwIkgR572MRUSa1Icfwtix0Lcv3Hgj0LNn9QVrmp7DGpMIRESajQsvhFWrQgeyNm2A8eN37DzQtm2Y3szU1Y/g8JpmAc3ojp0iks8mTYLJk0Mdf3hlrVd544FrrgnNQT17hgLN8IYEdd2q8rna3pxExzLdqlJEMqmsDA45JDQJvfBCGGW0OartVpV1jTVUY0VvZjoiEJGcVlERzgts2QJ//nPzTQJ1adA5AguGmtk9QFlMMYmINInbboNnnoHf/hZ69046muTUKxGYWZGZ/QFYAjwGvEDoZCYikpMWLIArroCTT4bzz086mmTVmgjM7EYzexcYD8wFDgPK3f0Bd1/bFAGKiGTa5s1w9tnQrh3cfTdYng+cU1eL2PnAO8Afgcfd/TMzq/nssohIDrj+enjlFXj4Ydh776SjSV5dTUP7ADcAw4H3zGwi0MbM8vSUiojkupkzQ4exsWPh1FOTjiY71HXV0DbgH8A/zGxX4GSgDVBmZs+6+1lNEKOISEZ88kkYUK5HD/j975OOJnvU1aHsi8Ayd/8gahbaDdgF+DvwfBPEJyKSMZdeCu+9B88/D7vvnnQ02aOupqG7gM0AZnYM8EvgAWAFMCLe0EREMueJJ+Cuu0IyOOaYpKPJLnW19bd09w+j52cCE9z9IeAhM3st3tBERDJj9Wo499zQg/j665OOJvvUmQjMrMDdtwJDgXENeK+ISOLc4TvfgbVr4amnYNddk44o+9RVmU8CZpjZamAj4f7FmFlvYF3MsYmINNrEieEy0Ztvhv79k44mO9V11dB4M3uGcBnpU759hLoWwPfjDk5EpDHefx8uuiicE/jRj5KOJnvVOcSEu89090fc/dOUae+4+yt1vdfMhpnZ22a20MyurKXcN8zMzazakfFERBpq2zb41rfC8wcegJYtk40nm8V2YxozawncDpwI9ANGm1m/asq1By4GZsUVi4jkn9/+Ngwr/Yc/QGFh0tFktzjvUDYQWOjui9x9MzCZ6i85vR74FbApxlhEJI/MmxfuJ/P1r28/KpCaxZkIugHLUl6XRdM+F90BrYe7/722BZnZODMrNbPS8vLyzEcqIs3GZ5+FAeU6doQJEzSgXH0kds9iM2sB/Ab4cV1l3X2Cuw9w9wFdunSJPzgRyVnXXgtz58I994Cqi/qJMxEsB3qkvO4eTavUHjgYeN7M3geKgGk6YSwiO+vFF+HXv4YLLgj3GZD6iTMRvAz0MbNeZrYLMAqYVjnT3de5e2d3L3T3QmAmcIq764bEItJgH38M3/wm9OoFv/lN0tHklth6B7v7VjO7CJgOtATudfc3zewXQKm7T6t9CSIi9XfJJbB0aTgqaNcu6WhyS6zDRLj7E8ATadOuraHskDhjEZHm69FH4b774Oqr4aijko4m9yR2slhEJBP++99wTuCww+BnP0s6mtykRCAiOcs93Hh+/XooLoZddkk6otykEURFJGfdcw/87W+hF3G/HcYtkPrSEYGI5KT33gsniI87Dn7wg6SjyW1KBCKSc7ZtC5eKFhTA/fdDC9VkjaKmIRHJOTffDP/5Tzgv0KNH3eWldsqjIpJTXn01DCMxciScdVbS0TQPSgQikjM2bQoDynXpAn/8owaUyxQ1DYlIzrj6apg/H/7xD9hzz6SjaT50RCAiOeGZZ8JlohdeCCeckHQ0zYsSgYhkvY8+grFj4YADwoliySwlAhHJPiUl4f6SLVpAYSHfP3kxK1fCxInQtm3SwTU/OkcgItmlpATGjYMNGwCYsmQgxUt6cd1pcxk4sH/CwTVPOiIQkexyzTWfJ4EV7MN3uZMvMpurS09LOLDmS4lARLLL0qUAOHAu97KJ1kzkHFotW5RsXM2YEoGIZJeePXHgV1zBdIbxay7jQN6Bnj2TjqzZ0jkCEckqiy/+Hd+5tD1PVwxlBI/yf9wRzhCPH590aM2WjghEJCts3RruNXzwT77OS7scw217/pSH+Qa2774wYQKMGZN0iM2WjghEJHGvvRZuMDNnDpx8MtxxRyt69LgeuD7p0PKCjghEJDEbN8JVV8GAAbBsGTz4IEybphFFm5qOCEQkEc89F7oLLFwI3/423HKLxg9Kio4IRKRJrV0bbjZ/3HFQUQH//Cfce6+SQJKUCESkSbjD1KnQty/cdx9cfjnMmwdDhyYdmahpSERit3x5GDX0scfgsMPgiSfg8MOTjkoq6YhARGJTUQF33gn9+sFTT4WRQ2fPVhLINjoiEJFYvPVWOBfwr3+F5p+77oL99086KqmOjghEJKM2b4brr4cvfAHefDOcD3j6aSWBbKYjAhHJmJkzQ8ewN9+EUaPgd7+DvfZKOiqpi44IRKTR1q+Hiy+Go46Cdevg8cdh0iQlgVyhIwIRaZQnnoDvfS/0DL7wQrjxRmjfPumopCF0RCAiO2XVKjjrLPja16BdO/j3v+HWW5UEclGsicDMhpnZ22a20MyurGb+j8xsvpnNNbNnzGzfOOMRkcZzhwceCB3DHnoIfv5zeOUVOPLIpCOTnRVbIjCzlsDtwIlAP2C0mfVLK/YqMMDd+wNTgZvjikdE6iHtpvGUlFSZvWgRfPWrMHZsSASvvQbXXgu77ppEsJIpcR4RDAQWuvsid98MTAZGpBZw9+fcfUP0cibQPcZ4RKQ2lTeNX7Ik7PYvWRJel5SwdSv8v/8HBx8Ms2bBHXfACy+EZCC5L85E0A1YlvK6LJpWk/OAJ2OMR0Rqk3LT+M9t2MBrl5VQVASXXgrHHw/z54eTwy10hrHZyIqv0szOBgYAv65h/jgzKzWz0vLy8qYNTiRfRDeNr7SR1lzJTQxYOY2yMpgyJYwV1F3H7c1OnIlgOZB6e4nu0bQqzOx44BrgFHf/rLoFufsEdx/g7gO6dOkSS7AieS/l5vDPMYT+zOVXXMnYdlNZsADOOAPMEoxPYhNnIngZ6GNmvcxsF2AUMC21gJkdBtxFSAKrYoxFRGrxyScw/cx7ubrgZo7mXxzHczjGM7uexJ/u3MYeeyQdocQptg5l7r7VzC4CpgMtgXvd/U0z+wVQ6u7TCE1B7YC/WtjVWOrup8QVk4gE69aFweBmzAiPOXNg27bjKGg5hAG7vMb1m3/Kj3tMoc1N1+qm8XnA3D3pGBpkwIABXlpamnQYIjnlww/hxRe3V/yvvRaGiG7VCgYNgsGDw+PII0PnMGl+zGyOuw+obp6GmBBphlatCpd3Vlb88+aF6a1bQ1ER/PSnoeIvKoI2bZKNVZKnRCDSDKxcub3SnzEDFiwI09u2DQPBjRwZKv6BA9X5S3akRCCSDUpKwnX8S5eGq3fGj6+1bX7ZsqoV/7vvhunt2sGXvgTf/Gao+I84AnbZpYk+g+QsJQKRpFX26K3szFXZoxdgzBjcYfHiqhX/+++H2R06wJe/HIoPHhzuB1yg/2ppIP1kRJKW1qPXgXc3dGPGD+Yx48lQ8ZeVhXmdOsExx8All4SK/5BDoGXLZMKW5kOJQCQhFRWhbX/xkh4s4hgW04v59ONFvsxKusKH8D9Pb7+iZ/DgcBN4De0gmaZEIBKjjz4KI3YuXhweqc/ffx8++wzgxc/L92QJQ3iewcxgcNeFHFj2jHrzSuyUCEQaYdOm0KRfU2X/0UdVy3fsCPvtF0bxHD48PO+1+Fl63foj9t30Fq2JRllp2xZungBKAtIElAhEalFRAcuXV1/JL1oEK1ZULb/rrmEY//32C52zevWKKvte4dGxY3VrOQ6+cFmDrhoSyST1LJa89+mn8Pbb8N57O1b4S5bA5s3by5qF0TcrK/bUSn6//WDvvdWGL9lJPYtFCOPrLFgQxtNPfSxZUrXcnu0+Y7+DduXQQ+HUU6tW9j17qkOWND9KBNLsrFmzY2U/f37VZpzWreGgg+Coru9z/n1U4FsAAAlQSURBVIoH6LtlLr1ZSC8Ws3vFNrhkgppmJG8oEUhOcof//rdqRV+5t78qZUDz3XYLl1wef3z4W/koLIyuvy8cAlvSDgk2ENrrlQgkTygRSFZzD52p0iv7+fNh7drt5Tp0CBX88OFVK/zu3etos0+7K1ed00WaISUCyQoVFeG6+vQ2/AULYP367eW6dAk3TD/zzKoV/t577+Tds3r23PEkQeV0kTyhRCBNbssWeP11mDUrPN54A956CzZu3F5mn31CBT927PbKvm/fkAgyavz4quP8QLiGf/z4DK9IJHspEUisKpt2Zs4Mj1mzwt2wNm0K8/duuYpDt83h2PZl9Dv/EPqdW0TfvjVdbx+DyvMAuoZf8pj6EUhGffoplJaGCr+y8l+5Msxr3RoOPzzcDKVoy4sMuvt8emx6Z3vn2bZtYYKu1hGJg/oRSCwqKkJHrNRK/403YNu2ML93bxg6NFT8gwZB//4pY+MXngOb0q/W2aCrdUQSoEQg9bZmTdVKf/bs0EkLwlU7gwbBKaeEin/gQOjcuZaF6WodkayhRJCP6nE3rM2bYe7c7e36M2fCwoVhXosWYe9+1Kjte/sHHtjAoRV0tY5I1lAiyDfV3A3LLxjHstVtmbnPqZ9X+nPmVA6RHK7gKSqCCy4If484InTUahRdrSOSNZQImlID70sbB7/6GlZu6MB8jmQORzCTImZuLOKDS/YBwgndI46Aiy4Ke/pFRaFTVsbHxNfVOiJZIz+uGiopYckVd7BiudOr+xb2uukS7OwmrnDS98Qh1qtkKipCy0tqB60FC2D+zHV8TIfPy/XhHYqYySBmU1R6G/37Q6tWGQ9HRBJW21VDzT8RRBXwrzZcxJX8CoA2bKBXt830OrTjDsMI9+oF7dvHEHhhYfVt4vvuu/1O5Dthy5YwfHJlhV/5N72D1l57RR2zSh+g7/rZ9GM+/ZlLJz7MSBwikt3y+/LR6Mbg5zCRQ5jHYnqxiP1Y/OH/snj5MF58ET7+uOpbOnfeMTlU/u3Zcyf3mBt5lcymTfDOO1Ur+wULwrQtW7aX69kzVPhDhmzvjdu3L+y5Z1SgpADG3a+2eRH5XPNPBFFF25WVdGXl9umbDF6twD0MXlbdrQZfeQUeeaRqRduiBfToUfONSfbaq4b29HpeJbN+fdibT6/wFy0KzT2VMey/f6jghw8Pf/v1C8Mqt2tXx/ZQ27yIpGn+TUONbJLZti2MY19doli0aHuv2Upt2mxPDFWOKBb8nV7Xn0v7jWGM5A/Zg/m7Hs6Cs29gfvuizyv9Zcu2L6tVq3BZZmVFX/m3T59wUldEpL50jiDGk7QbN1a9eXl6wkhvdurU4kNaVmxhFXtVCeegg6pW9n37hr3+guZ/zCYiTSC/zxHE3BTSpk2oxA86aMd5lc1OVY8i9mTr1qoVfs+eus+tiCSn+R8RiIhIrUcE2g8VEclzsSYCMxtmZm+b2UIzu7Ka+bua2YPR/FlmVhhnPCIisqPYEoGZtQRuB04E+gGjzaxfWrHzgLXu3hv4LUQ9vkREpMnEeUQwEFjo7ovcfTMwGRiRVmYE8ED0fCow1Czjo9qIiEgt4kwE3YCUq+Ipi6ZVW8bdtwLrgE7pCzKzcWZWamal5eXlMYUrIpKfcuJksbtPcPcB7j6gS8bvXi4ikt/iTATLgR4pr7tH06otY2YFQAdgTYwxiYhImjgTwctAHzPrZWa7AKOAaWllpgHfip6fDjzrudaxQUQkx8XaoczMTgJ+B7QE7nX38Wb2C6DU3aeZWWtgInAY8CEwyt0X1bHMcqCawYNySmdgddJBZBFtj+20LarS9qiqMdtjX3evtm0953oWNwdmVlpTD798pO2xnbZFVdoeVcW1PXLiZLGIiMRHiUBEJM8pESRjQtIBZBltj+20LarS9qgqlu2hcwQiInlORwQiInlOiUBEJM8pETQhM+thZs+Z2Xwze9PMLk46pqSZWUsze9XM/pZ0LEkzs45mNtXM3jKzBWZ2ZNIxJcnMfhj9n7xhZpOifkd5wczuNbNVZvZGyrQ9zexpM3s3+rtHptanRNC0tgI/dvd+QBFwYTVDc+ebi4EFSQeRJX4P/MPdDwK+QB5vFzPrBvwAGODuBxM6pY5KNqomdT8wLG3alcAz7t4HeCZ6nRFKBE3I3Ve6+yvR8/WEf/T0EVnzhpl1B74G/CnpWJJmZh2AY4B7ANx9s7t/lGxUiSsA2kTjkLUFViQcT5Nx9xcIoy2kSh22/wHg65lanxJBQqK7sR0GzEo2kkT9DrgcqEg6kCzQCygH7ouayv5kZrslHVRS3H05cAuwFFgJrHP3p5KNKnF7ufvK6PkHwF6ZWrASQQLMrB3wEHCJu3+cdDxJMLOTgVXuPifpWLJEAXA48Ed3Pwz4lAwe+ueaqP17BCFBdgV2M7Ozk40qe0SDc2bs2n8lgiZmZq0ISaDE3R9OOp4EHQ2cYmbvE+5ed5yZFScbUqLKgDJ3rzxCnEpIDPnqeGCxu5e7+xbgYeCohGNK2n/NbB+A6O+qTC1YiaAJRbfhvAdY4O6/STqeJLn7Ve7e3d0LCScBn3X3vN3jc/cPgGVmdmA0aSgwP8GQkrYUKDKzttH/zVDy+OR5JHXY/m8Bj2VqwUoETeto4BzC3u9r0eOkpIOSrPF9oMTM5gKHAjcmHE9ioiOjqcArwDxCXZU3w02Y2STgJeBAMyszs/OAXwJfMbN3CUdMv8zY+jTEhIhIftMRgYhInlMiEBHJc0oEIiJ5TolARCTPKRGIiOQ5JQKRNGb2STXTrjOz5SmX/b5mZl3NbI2Z7Z5W9lEzO9PMxppZedp7+plZYeqokiJJUyIQqb/fuvuhKY8VwHTg1MoC0eBxXwIejyY9mPaefO4kJllKiUCkcSZRdXjkU4Hp7r4hoXhEGkyJQKT+fpjSxPNcNG06cLiZdYpejyIkh0pnpjUNtWnSiEXqoSDpAERyyG/d/ZbUCe6+2cymAaeb2UOEocWnpxR50N0vSn1PGDpHJHsoEYg03iTgp4ABj0WjZYrkDDUNiTTe80Af4EKqNguJ5AQlApEdtY1GfKx8/Cia/sO09v5CAHevIIyU2QmYkbas9HMElWPqH5i2jjOa4oOJVEejj4qI5DkdEYiI5DklAhGRPKdEICKS55QIRETynBKBiEieUyIQEclzSgQiInnu/wOCyGawUj4G5AAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "atPu3hldrtuw"
      },
      "source": [
        "#Predicting a new result with Linear Regression"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Z6qwFcSqr1U1",
        "outputId": "086266dc-0c90-400a-e606-bb06d652ba11"
      },
      "source": [
        "lin_reg.predict([[6.5]])"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([330378.78787879])"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ur-9w2GXtwG8"
      },
      "source": [
        "#Predicting a new result with Polynomial Regression "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "u0c6TuqFt2LI",
        "outputId": "abbc9fba-1c92-4585-f733-10c9fdf61fa5"
      },
      "source": [
        "lin_reg2.predict(polyreg.fit_transform([[6.5]]))"
      ],
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([158862.45265155])"
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hGfgQOGstGOu",
        "outputId": "da56efaa-3d00-4424-cc47-4dad92c6bc2f"
      },
      "source": [
        "lin_reg2.predict(polyreg.fit_transform([[6]]))"
      ],
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([143275.05827509])"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lpfsazyC_Q-R",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "680f7cab-ae6e-4faf-bd14-a4d6dba3fbf7"
      },
      "source": [
        "lin_reg2.predict(polyreg.fit_transform([[8.9]]))"
      ],
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([496488.97013401])"
            ]
          },
          "metadata": {},
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KpJFreMl4Ez-",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b92e7b46-60b8-4b38-9376-31fc6e1f26b8"
      },
      "source": [
        "lin_reg2.predict(polyreg.fit_transform([[9.0]]))"
      ],
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([528694.63869462])"
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ]
    }
  ]
}