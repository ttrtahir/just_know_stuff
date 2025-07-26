### Summary in one line

Automatic Differentiation = **"Doing the chain rule step-by-step inside the computer, automatically, without writing the derivatives yourself."**

## Prior knowledge

Differentiation has 4 types:
- Manual
- Numerical
- Symbolic
- Automatic

Manual is when you calculate it one by one, in coding this corresponds to brute forcing. 

Symbolic is when we automate the Manual option with set of rules to find the differentiation, this allows us to find the exact derivative and bypasses the rounding or truncating errors from Numerical but it has also some downsides,  "[[Expression Swell]]" for example on [[Soft ReLU]], the derivative expression becomes exponentially longer than the original function due to some derivative rules such as product rule.

Numerical a.k.a. finite differences, approximate derivatives. Limit definition of the derivative is the simplest way to explain this: partial derivative of a function f with respect to x, with h's. But this one only calculates secant and not tangent, the closer the h is the closer the secant will be to tangent. This is not precise but an approximation, it will cause rounding error and then truncation error, the balance between them is based on the calculators preference.

## ELI5

Imagine you have a recipe that takes some ingredients (numbers) and gives you a cake (a final number). But you also want to know:

**"If I add a little more flour, how much will the cake change?"**

This is what differentiation does — it tells you how sensitive the output is to small changes in the input.

- **Automatic Differentiation (AD)** is like having a super-smart calculator that **follows every step of the recipe**, keeps track of how each ingredient affects the next step, and finally gives you the exact sensitivity (derivative) **automatically**.

It’s not guessing, and it’s not symbolically solving math — it just keeps track of how each operation changes the result.

### Difference Table

| **Type**                                           | **How It Works (ELI5)**                                                                                                                        | **Good/Bad For**                                                                                            |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Symbolic Differentiation**                       | Like solving derivatives by hand with calculus rules (e.g., derivative of `x²` is `2x`). Computer algebra systems (like WolframAlpha) do this. | ✅ Perfect for simple math expressions ❌ Becomes super messy and slow for deep neural networks               |
| **Numerical Differentiation (Finite Differences)** | You poke the system a little: change `x` by a tiny amount (`x + h`) and see how much the output changes. Formula: `(f(x+h) - f(x)) / h`.       | ✅ Easy to understand ❌ Very inaccurate for tiny `h`, and slow (you need to compute the function many times) |
| **Automatic Differentiation (AD)**                 | Follows the recipe step by step, applying the chain rule automatically. Works by storing intermediate results during the calculation.          | ✅ Fast and very accurate, even for huge neural networks ❌ Needs special implementation                      |

![[Pasted image 20250726010352.png]]
(source: https://arxiv.org/pdf/1502.05767)

### Two parts of AD

- **Forward Mode AD** – Tracks how each input changes the output step by step (good when you have **few inputs, many outputs**).

- **Reverse Mode AD** (used in deep learning) – Works backward, starting from the output and figuring out how each input affected it (good when you have **many inputs, one output**, like loss in a neural network).

## Proper definition

**Automatic Differentiation (AD)** is a set of techniques to **compute exact derivatives of functions expressed as computer programs by systematically applying the chain rule of calculus to every elementary operation in the computation graph**.

### Formally

If we have a function f : Rn → Rm that is implemented as a sequence of elementary operations (addition, multiplication, trigonometric functions, etc.), AD evaluates derivatives by:

1. **Decomposing** f into a computational graph of primitive operations.
2. **Propagating derivatives** through this graph using the **chain rule**:
	
	   ∂y/∂x = ∂y/∂u * ∂u/∂x
    
    for every intermediate variable u.
3. Producing derivatives that are **exact up to machine precision** (unlike numerical differentiation).

### **Forward vs Reverse Mode**

- **Forward Mode AD**: Propagates **directional derivatives** alongside the value computation.  
    For each intermediate variable vi:
    
      dot(vi) = v˙i = sum_j ∑ ∂vi/∂vj * v˙j 
    
    Good for n≪m (few inputs, many outputs). Fills columns first.
    
- **Reverse Mode AD**: Propagates **adjoints (sensitivities)** backward after a forward pass.  
    For each intermediate variable vi:
    
      bar(vj) = vˉj = sum_i ∑ ∂vi/∂vj * vˉi 
    
    Good for n≫m (many inputs, single output) → this is essentially **backpropagation in neural networks**. Fills rows first.