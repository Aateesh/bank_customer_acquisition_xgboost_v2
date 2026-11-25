import joblib

MODEL_PATH = "xgb_customer_acquisition_pipeline.joblib"

print("=" * 60)
print("MODEL INSPECTION")
print("=" * 60)

model = joblib.load(MODEL_PATH)

print("\nModel type:")
print(type(model))

print("\nModel:")
print(model)

print("\nModel attributes:")

if hasattr(model, "named_steps"):
    print("\nPipeline steps:")
    for name, step in model.named_steps.items():
        print(f"  {name}: {type(step)}")

if hasattr(model, "feature_names_in_"):
    print("\nFeature names:")
    print(model.feature_names_in_)

if hasattr(model, "n_features_in_"):
    print("\nNumber of input features:")
    print(model.n_features_in_)

if hasattr(model, "classes_"):
    print("\nClasses:")
    print(model.classes_)

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)