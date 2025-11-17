class ResponseCode:
    SUCCESS = 200
    BAD_REQUEST = 400
    NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500


class RequestMethod:
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"


class RAGResponseStatus:
    SUCCESS = "success"
    INTERNAL_SERVICE_ERROR = "internal service error"
    NO_KNOWLEDGE_REQUIRED = "no knowledge required"
    LACK_OF_KNOWLEDGE = "lack of knowledge"


HTTP_HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
}
