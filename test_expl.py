from src.tools.explanation_tool import ExplanationTool

explainer = ExplanationTool()

test_state = [1, 0, 0.002, 0.3, 0.6]
test_action = 4
test_goal = "EMERGENCY_FIRE"

response = explainer.explain(test_state, test_action, test_goal)

print("\nGemma Response:\n")
print(response)