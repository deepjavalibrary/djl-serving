# Properties manager
DJLServing gives the users a flexibility to write their own handlers but also comes with built-in default handlers for various optimization frameworks. Each framework offers a distinct set of properties that users can customize to fine-tune the optimization process. These properties can be configured either in the serving.properties file or through environment variables. Refer to [this document](../../../../../serving/docs/lmi/lmi_environment_variable_instruction.md) for instructions on configuring properties for each handler.

This module leverages pydantic v1 to manage a common set of properties and handler-specific properties. The following guide outlines the process of adding new properties for the default handlers.

## Instruction on adding new properties
1. Update Pydantic Class:
   - Identify whether the property is common or handler-specific, and add it to the respective pydantic class. 
   - If the property is optional, ensure it is marked as Optional or assign a default value if applicable. 
   - Implement validation for the new property using @validator for field validation or @root_validator for global validation, especially when the property depends on others. 
   - Minimize adding properties validation in the handler; prefer handling everything in the properties pydantic classes.
2. Unit testing: 
   - Include a unit test case to verify proper setting of the new property. 
   - Test both positive and negative cases to ensure robust validation.
3. Documentation Update:
   - Add the property with their details to [configurations_large_model_inference_containers.md](../../../../../serving/docs/lmi/configurations_large_model_inference_containers.md). Provide comprehensive information about the purpose and usage of the added property.

