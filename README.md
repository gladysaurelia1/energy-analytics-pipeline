## Machine Learning Insights

### Overfitting analysis

During development, I encountered and resolved a critical overfitting issue. The initial model achieved 99.44% R² by using lag features that caused data leakage. 

**See detailed analysis:** `experiments/overfitting_demo/`

**Key Learning:** High metrics don't always indicate a good model. The production version achieves realistic 75-80% accuracy and actually generalizes to unseen data.

This demonstrates:
- Critical evaluation of model performance
- Understanding of data leakage pitfalls  
- Production-ready ML engineering practices
- Ability to debug and improve models!!!!! 🤩