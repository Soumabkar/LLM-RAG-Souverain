import asyncio
from Engine.models import team, user

async def main():
    await team("DATA-01", "Data Platform").create_team()
    await user("ksoumahoro", "TonMotDePasse", "karim@entreprise.fr").create_user("DATA-01")

asyncio.run(main())