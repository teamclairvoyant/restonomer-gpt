import { MouseEvent, MouseEventHandler } from "react";
import { Heading, Link, Card, CardHeader, Flex, Spacer } from "@chakra-ui/react";
import { ExternalLinkIcon } from '@chakra-ui/icons'

export function EmptyState(props: {
  onChoice: (question: string) => any
}) {
  const handleClick = (e: MouseEvent) => {
    props.onChoice((e.target as HTMLDivElement).innerText);
  }
  return (
    <div className="p-8 rounded flex flex-col items-center">
      <Heading fontSize="6xl" fontWeight={"bold"} mb={1} color={"green.700"}>Restonomer GPT</Heading>
      <Heading fontSize="l" fontWeight={"normal"} mb={1} color={"gray.500"} marginTop={"10px"} textAlign={"center"}>Ask me anything about Restonomer&apos;s{" "}
      <Link href='https://teamclairvoyant.github.io/restonomer/docs/restonomer_intro' color={"blue.500"}>
        Documentation
      </Link></Heading>
      <Flex marginTop={"25px"} grow={1} maxWidth={"800px"}>
        <Card onMouseUp={handleClick} width={"48%"}  backgroundColor={"rgb(214, 245, 214)"} _hover={{"background-color": "rgb(78,78,81)"}} cursor={"pointer"} justifyContent={"center"}>
          <CardHeader justifyContent={"center"}>
            <Heading fontSize="lg" fontWeight={"medium"} mb={1} color={"green.500"} textAlign={"center"}>What is Restonomer ?</Heading>
          </CardHeader>
        </Card>
        <Spacer />
        <Card onMouseUp={handleClick} width={"48%"}  backgroundColor={"rgb(214, 245, 214)"} _hover={{"background-color": "rgb(78,78,81)"}} cursor={"pointer"} justifyContent={"center"}>
          <CardHeader justifyContent={"center"}>
            <Heading fontSize="lg" fontWeight={"medium"} mb={1} color={"green.500"} textAlign={"center"}>How can I integrate Restonomer in my existing application ?</Heading>
          </CardHeader>
        </Card>
      </Flex>
      <Flex marginTop={"25px"} grow={1} maxWidth={"800px"}>
        <Card onMouseUp={handleClick} width={"48%"}  backgroundColor={"rgb(214, 245, 214)"} _hover={{"background-color": "rgb(78,78,81)"}} cursor={"pointer"} justifyContent={"center"}>
          <CardHeader justifyContent={"center"}>
            <Heading fontSize="lg" fontWeight={"medium"} mb={1} color={"green.500"} textAlign={"center"}>What are different transformations supported by Restonomer ?</Heading>
          </CardHeader>
        </Card>
        <Spacer />
        <Card onMouseUp={handleClick} width={"48%"}  backgroundColor={"rgb(214, 245, 214)"} _hover={{"background-color": "rgb(78,78,81)"}} cursor={"pointer"} justifyContent={"center"}>
          <CardHeader justifyContent={"center"}>
            <Heading fontSize="lg" fontWeight={"medium"} mb={1} color={"green.500"} textAlign={"center"}>What is the checkpoint configuration ?</Heading>
          </CardHeader>
        </Card>
      </Flex>
    </div>
  );
}