import asyncio

import utils




async def main():
    wrapper = utils.wrappers.PropertyWrapper()
    data = await wrapper.prematch(team1=15, team2=7119388)
    data = utils.nn.tools.batch_to_tensor(data)
    data = utils.nn.tools.batch_to_device(data, 'cuda')
    
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
