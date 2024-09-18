from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)


chain = RunnableParallel(
    passed=RunnablePassthrough(),
    modified=lambda x: x["num"] + 1,
)

print(chain.invoke({"num": 1}))
